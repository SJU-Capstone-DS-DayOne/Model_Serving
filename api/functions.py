import os
import pymysql
import pandas as pd
import ast
import torch
from dotenv import  load_dotenv


load_dotenv()  # .env 파일을 불러와 환경 변수 설정

MODEL_HOST = os.getenv('MODEL_HOST')
MODEL_PORT = int(os.getenv('MODEL_PORT'))
MODEL_USER = os.getenv('MODEL_USER')
MODEL_PW = os.getenv('MODEL_PW')
DB = os.getenv('DB')


# SQL Database에 연결
def DB_CONNECT():
    connection = pymysql.connect(
        host = MODEL_HOST,
        port = MODEL_PORT,
        user = MODEL_USER,
        passwd = MODEL_PW,
        db = DB,
        charset = 'utf8mb4'
    )

    return connection

def DATA_LOADER(dataset):
    # 현재 파일(main.py)의 디렉토리 경로를 가져옵니다.
    current_dir = os.path.dirname(__file__)

    # 'Data' 폴더 경로 설정
    data_folder = os.path.join(current_dir, '..', 'Data')

    if dataset == 'user':
        # 유저/레스토랑 임베딩 로드
        user_path = os.path.join(data_folder, 'user_embedding.csv')
        user_embedding = pd.read_csv(user_path, index_col='user_id')

        # string 형식을 python list로 변환
        user_embedding['embedding'] = user_embedding['embedding'].apply(ast.literal_eval)

        return user_embedding

    else:
        if dataset == 'KJ':
            # 광진 레스토랑 임베딩 로드
            restaurant_path = os.path.join(data_folder, 'rst_embedding.csv')
            restaurant_embedding_KJ = pd.read_csv(restaurant_path, index_col="user_id")
            restaurant_embedding_KJ["embedding"] = restaurant_embedding_KJ["embedding"].apply(ast.literal_eval)

            return restaurant_embedding_KJ

        elif dataset == 'HD':
            # 홍대 레스토랑 임베딩 로드
            restaurant_path = os.path.join(data_folder, 'rst_embedding_HD.csv')
            restaurant_embedding_HD = pd.read_csv(restaurant_path, index_col="user_id")
            restaurant_embedding_HD["embedding"] = restaurant_embedding_HD["embedding"].apply(ast.literal_eval)

            return restaurant_embedding_HD
        else:
            # 잠실 레스토랑 임베딩 로드
            restaurant_path = os.path.join(data_folder, 'rst_embedding.csv')
            restaurant_embedding_JS = pd.read_csv(restaurant_path, index_col="user_id")
            restaurant_embedding_JS["embedding"] = restaurant_embedding_JS["embedding"].apply(ast.literal_eval)

            return restaurant_embedding_JS

def SAVE(row):
    # 레스토랑 임베딩 경로 설정 및 로드
    current_dir = os.path.dirname(__file__)
    data_folder = os.path.join(current_dir, '..', 'Data')
    user_path = os.path.join(data_folder, 'user_embedding.csv')
    
    user_embedding = pd.read_csv(user_path)
    
    try:
        user_embedding.drop(columns='Unnamed: 0',inplace=True)
    except:
        pass

    user_embedding.loc[len(user_embedding)] = row
    user_embedding.to_csv(user_path)
    
"""
    try:
        user_embedding.drop(columns='Unnamed: 0',inplace=True)
        restaurant_embedding_KJ.drop(columns='Unnamed: 0',inplace=True)
    except:
        pass

    return user_embedding, restaurant_embedding_KJ, restaurant_embedding_HD, restaurant_embedding_JS

"""