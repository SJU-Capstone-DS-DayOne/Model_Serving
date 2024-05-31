import uvicorn
from fastapi import FastAPI
import json
import torch
from torch import nn
import numpy as np
from . import functions
import pymysql


# DB에 연결
cursor = functions.DB_CONNECT()


# 유저/레스토랑 임베딩 텐서 로드
user_embedding, restaurant_embedding_KJ, restaurant_embedding_HD, restaurant_embedding_JS = functions.DATA_LOADER()

# 전체 레스토랑에 대한 임베딩 텐서화
embeddings_list = restaurant_embedding_KJ['embedding'].tolist()
embeddings_tensor_KJ = torch.tensor(embeddings_list)

embeddings_list = restaurant_embedding_HD['embedding'].tolist()
embeddings_tensor_HD = torch.tensor(embeddings_list)

embeddings_list = restaurant_embedding_JS['embedding'].tolist()
embeddings_tensor_JS = torch.tensor(embeddings_list)


# 활성 함수 정의
activate_f = nn.Sigmoid()





""" API """
app = FastAPI()

# Default Screen
@app.get("/")
def root():
    return {"Default": "Hello World-!"}


# 단일 유저 추천
@app.get("/recommend")
async def recommend(user: int):
    # 유저 임베딩 텐서화
    user_embedding_vector = torch.tensor(user_embedding.loc[user,'embedding'])
    
    # 단일 유저 추론
    one_user_predict = activate_f(torch.matmul(user_embedding_vector,embeddings_tensor.t()))
    _, one_user_rating = torch.topk(one_user_predict, k=100)
    result_restaurant_list = np.array(one_user_rating).tolist()

    result_dict = {
        "user_num": user,
        "result": result_restaurant_list
    }

    return json.dumps(result_dict)


# 커플 유저 추천
@app.get("/recommend/couple")
async def recommend_couple(user1: int, user2: int, district: str):
    # 각 유저의 임베딩 텐서화
    user1_embedding_vector = torch.tensor(user_embedding.loc[user1,'embedding'])
    user2_embedding_vector = torch.tensor(user_embedding.loc[user2,'embedding'])

    # 커플 임베딩 생성
    # couple_embedding_vector = torch.mul(user1_embedding_vector, user2_embedding_vector)
    couple_embedding_vector = 0.5 * user1_embedding_vector + 0.5 * user2_embedding_vector
    
    # 커플 유저 결과 추론 (음식점/카페/술집 전체 id 리스트)
    # 지역별로 달리하여
    if district == '광진':
        couple_user_predict = activate_f(torch.matmul(couple_embedding_vector, embeddings_tensor_KJ.t()))
    elif district == '홍대':
        couple_user_predict = activate_f(torch.matmul(couple_embedding_vector, embeddings_tensor_HD.t()))
    elif district == '잠실':
        couple_user_predict = activate_f(torch.matmul(couple_embedding_vector, embeddings_tensor_JS.t()))
    else:
        return "[Error] Wrong District Name"


    _, couple_user_rating = torch.topk(couple_user_predict,k=1000)
    result_restaurant_list = np.array(couple_user_rating).tolist()

    # sql문에 넣을 id들의 집합으로 변환
    ids_query = ', '.join(map(str, result_restaurant_list))

    # 결과값에 매칭되는 RST/CAFE/BAR인지의 types 리스트 요청
    sql = f"SELECT name, type FROM restaurant WHERE restaurant_id IN ({ids_query}) ORDER BY FIELD(restaurant_id, {ids_query})"
    cursor.execute(sql)
    types = cursor.fetchall()

    # 결과값 음식점/카페/술집 분리
    RST, CAFE, BAR = [], [], []
    for result_id, type in zip(result_restaurant_list, types):
        category = type[1]

        if category == 'RST':
            RST.append(result_id)
        elif category == 'CAFE':
            CAFE.append(result_id)
        else:
            BAR.append(result_id)


    # 결과값 딕셔너리화
    result_dict = {
        "RST": RST[:18],
        "CAFE": CAFE[:18],
        "BAR": BAR[:18]
    }

    return result_dict


# Test
@app.get("/test/couple")
async def recommend_couple(user1: int, user2: int):
    # 각 유저의 임베딩 텐서화
    user1_embedding_vector = torch.tensor(user_embedding.loc[user1,'embedding'])
    user2_embedding_vector = torch.tensor(user_embedding.loc[user2,'embedding'])

    # 커플 임베딩 생성
    couple_embedding_vector = 0.5 * user1_embedding_vector + 0.5 * user2_embedding_vector
    
    # 커플 유저 결과 추론 (음식점/카페/술집 전체 id 리스트)
    couple_user_predict = activate_f(torch.matmul(couple_embedding_vector, embeddings_tensor_KJ.t()))


    _, couple_user_rating = torch.topk(couple_user_predict,k=50)
    result_restaurant_list = np.array(couple_user_rating).tolist()


    # 유저의 인터액션 불러오기
    sql = f"SELECT RST.name FROM review AS REV INNER JOIN restaurant AS RST ON REV.restaurant_id = RST.restaurant_id WHERE REV.member_id = {user1};"
    cursor.execute(sql)
    user1_interactions = cursor.fetchall()
    user1_interactions = [user_interaction[0] for user_interaction in user1_interactions]

    sql = f"SELECT RST.name FROM review AS REV INNER JOIN restaurant AS RST ON REV.restaurant_id = RST.restaurant_id WHERE REV.member_id = {user2};"
    cursor.execute(sql)
    user2_interactions = cursor.fetchall()
    user2_interactions = [user_interaction[0] for user_interaction in user2_interactions]

    # sql문에 넣을 id들의 집합으로 변환
    ids_query = ', '.join(map(str, result_restaurant_list))

    # 결과값에 매칭되는 RST/CAFE/BAR인지의 types 리스트 요청
    sql = f"SELECT name, type FROM restaurant WHERE restaurant_id IN ({ids_query}) ORDER BY FIELD(restaurant_id, {ids_query})"
    cursor.execute(sql)
    sql_results = cursor.fetchall()

    # 결과값 음식점/카페/술집 분리
    RST, CAFE, BAR = [], [], []
    for result_id, sql_result in zip(result_restaurant_list, sql_results):
        restaurant_name = sql_result[0]
        category = sql_result[1]

        if category == 'RST':
            RST.append((result_id, restaurant_name))
        elif category == 'CAFE':
            CAFE.append((result_id, restaurant_name))
        else:
            BAR.append((result_id, restaurant_name))


    # 결과값 딕셔너리화
    result_dict = {
        "User1 Records": user1_interactions,
        "User2 Records": user2_interactions,
        "RST": RST,
        "CAFE": CAFE,
        "BAR": BAR
    }

    return result_dict


# main
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)