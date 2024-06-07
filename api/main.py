import uvicorn
from fastapi import FastAPI
import json
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from . import functions
import pymysql
from typing import List, Union
from pydantic import BaseModel


# DB에 연결
connection = functions.DB_CONNECT()
cursor = connection.cursor()

# 유저/레스토랑 임베딩 텐서 로드
# user_embedding, restaurant_embedding_KJ, restaurant_embedding_HD, restaurant_embedding_JS = functions.DATA_LOADER()



#  --
user_embedding = functions.DATA_LOADER('user')
restaurant_embedding_KJ = functions.DATA_LOADER('KJ')
restaurant_embedding_HD = functions.DATA_LOADER('HD')
restaurant_embedding_JS = functions.DATA_LOADER('JS')

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
    one_user_predict = activate_f(torch.matmul(user_embedding_vector,embeddings_tensor_KJ.t()))
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
    # DB 연결 재확인
    connection.ping(reconnect=True)

    # 각 유저의 임베딩 텐서화
    user1_embedding_vector = torch.tensor(user_embedding[user_embedding.index == user1]['embedding'].values[0])
    user2_embedding_vector = torch.tensor(user_embedding[user_embedding.index == user2]['embedding'].values[0])

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
    # sql = f"SELECT name, type FROM restaurant WHERE restaurant_id IN ({ids_query}) ORDER BY FIELD(restaurant_id, {ids_query})"
    sql_query = f"""
                    SELECT r.restaurant_id, r.type
                    FROM restaurant AS r
                    WHERE r.restaurant_id IN ({ids_query})
                    AND r.restaurant_id IN (
                        SELECT DISTINCT restaurant_id
                        FROM menu
                        WHERE ranking IS NOT NULL
                    )
                    ORDER BY FIELD(r.restaurant_id, {ids_query});
                """
    cursor.execute(sql_query)
    sql_results = cursor.fetchall()

    # 결과값 음식점/카페/술집 분리
    RST, CAFE, BAR = [], [], []
    for sql_result in sql_results:
        restaurant_id = sql_result[0]
        category = sql_result[1]

        if category == 'RST':
            RST.append(restaurant_id)
        elif category == 'CAFE':
            CAFE.append(restaurant_id)
        else:
            BAR.append(restaurant_id)


    # 결과값 딕셔너리화
    result_dict = {
        "RST": RST[:18],
        "CAFE": CAFE[:18],
        "BAR": BAR[:18]
    }

    return result_dict


# Test
@app.get("/test/couple")
async def test_couple(user1: int, user2: int, district: str):
    # 각 유저의 임베딩 텐서화
    user1_embedding_vector = torch.tensor(user_embedding[user_embedding.index == user1]['embedding'].values[0])
    user2_embedding_vector = torch.tensor(user_embedding[user_embedding.index == user2]['embedding'].values[0])

    # 커플 임베딩 생성
    couple_embedding_vector = 0.5 * user1_embedding_vector + 0.5 * user2_embedding_vector
    
    # 지역별로 달리하여
    # 커플 유저 결과 추론 (음식점/카페/술집 전체 id 리스트)
    if district == '광진':
        couple_user_predict = activate_f(torch.matmul(couple_embedding_vector, embeddings_tensor_KJ.t()))
        _, couple_user_rating = torch.topk(couple_user_predict,k=1000)
        result_restaurant_list = np.array(couple_user_rating).tolist()
        # result_restaurant_list = [r+]
    elif district == '홍대':
        couple_user_predict = activate_f(torch.matmul(couple_embedding_vector, embeddings_tensor_HD.t()))
        _, couple_user_rating = torch.topk(couple_user_predict,k=500)
        result_restaurant_list = np.array(couple_user_rating).tolist()
    elif district == '잠실':
        couple_user_predict = activate_f(torch.matmul(couple_embedding_vector, embeddings_tensor_JS.t()))
        _, couple_user_rating = torch.topk(couple_user_predict,k=500)
        result_restaurant_list = np.array(couple_user_rating).tolist()
    else:
        return "[Error] Wrong District Name"


    # _, couple_user_rating = torch.topk(couple_user_predict,k=1000)
    # result_restaurant_list = np.array(couple_user_rating).tolist()


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
    sql_query = f"""
                    SELECT r.restaurant_id, r.name, r.type
                    FROM restaurant AS r
                    WHERE r.restaurant_id IN ({ids_query})
                    AND r.restaurant_id IN (
                        SELECT DISTINCT restaurant_id
                        FROM menu
                        WHERE ranking IS NOT NULL
                    )
                    ORDER BY FIELD(r.restaurant_id, {ids_query});
                """
    cursor.execute(sql_query)
    sql_results = cursor.fetchall()

    # 결과값 음식점/카페/술집 분리
    RST, CAFE, BAR = [], [], []
    for sql_result in sql_results:
        restaurant_id = sql_result[0]
        restaurant_name = sql_result[1]
        category = sql_result[2]

        if category == 'RST':
            RST.append((restaurant_id, restaurant_name))
        elif category == 'CAFE':
            CAFE.append((restaurant_id, restaurant_name))
        else:
            BAR.append((restaurant_id, restaurant_name))


    # 결과값 딕셔너리화
    result_dict = {
        "User1 Records": user1_interactions,
        "User2 Records": user2_interactions,
        "RST": RST[:18],
        "CAFE": CAFE[:18],
        "BAR": BAR[:18]
    }

    return result_dict


class Item(BaseModel):
    restaurantids: List[int]

# Cold Start
@app.post("/coldstart")
async def coldstart(new_user: int, item: Item):
    global restaurant_embedding_KJ, user_embedding

    # 신규 유저 임베딩 초기화
    new_user_embedding = torch.empty(1, 64)
    nn.init.normal_(new_user_embedding, std=0.1)
    new_user_embedding = new_user_embedding[0]

    # post로 전달받은 신규 유저가 선택한 5개의 식당 id
    ids = item.restaurantids
    # 선택한 식당에 맞게 임베딩 업데이트
    for id in ids:
        selected_embedding = torch.tensor(restaurant_embedding_KJ[restaurant_embedding_KJ.index == id]['embedding'].values[0])
        new_user_embedding += selected_embedding
    
    # 정규화
    new_user_embedding = new_user_embedding / len(ids)
    # print(new_user_embedding)

    # 새로 추가할 신규 유저 행 생성
    update_row = [new_user, new_user_embedding.tolist()]


    # 신규 유저 업데이트 후 임베딩 다시 불러오기
    functions.SAVE(update_row)
    user_embedding = functions.DATA_LOADER('user')

    return ids

# Review Sorting
@app.get("/review/sort")
async def sort(user_id: int, restaurant_id: int):
    global user_embedding

    # DB 연결 재확인
    connection.ping(reconnect=True)

    cur_user_embedding = torch.tensor(user_embedding[user_embedding.index == user_id]['embedding'].values[0])
    
    # 주어진 restaurant_id의 모든 review_id, member_id 조회 (로그인된 user_id는 제외)
    sql_query = f"""
                SELECT review_id, member_id, content
                FROM review
                WHERE restaurant_id = {restaurant_id}
                AND member_id != {user_id};
                """
    cursor.execute(sql_query)
    review_datas = cursor.fetchall()

    # sql결과에 대해 리뷰 작성한 user와 현재 user의 cosine similarity 계산
    compare_list = []
    for review_data in review_datas:
        review_id, member_id, review_content = review_data[0], review_data[1], review_data[2]
        reviewer_embedding = torch.tensor(user_embedding[user_embedding.index == member_id]['embedding'].values[0])
        similarity = F.cosine_similarity(cur_user_embedding, reviewer_embedding, dim=0).item()
        compare_list.append((review_id, member_id, similarity, review_content))
    
    # similarity를 기준으로 내림차순 정렬
    sorted_list = sorted(compare_list, key=lambda x: x[2], reverse=True)

    # review_id만 추출하여 result 리스트에 할당
    result = [item[0] for item in sorted_list]

    result_dic = {
        "review_ids": result
    }

    return result_dic




# main
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)