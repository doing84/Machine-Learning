import os
import sys
import pandas as pd
from youtube_search_and_extract import youtube_search, get_video_details, get_video_comments, get_channel_subscriber_count
import logging
from googleapiclient.errors import HttpError

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('youtube_api.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# API 키 설정
api_key = 'AIzaSyAOYpm2_Po0dwFTsKHuM0WMMmKmX08NhFk'

# 검색어 리스트 설정
search_queries = ['나이키 에어포스1 07 화이트', '나이키 덩크 로우 레트로 블랙', "나이키 에어포스 1 '07 WB 플랙스", "나이키 줌 보메로 5 코블스톤 앤 플랫 퓨터", "뉴발란스 530 스틸 그레이", 
                  "뉴발란스 530 화이트", "뉴발란스 993 메이드 인 USA 그레이 - D 스탠다드", "아디다스 삼바 OG 클라우드 화이트", "아디다스 삼바 OG 블랙 화이트 검", 
                  "아디다스 복서 호그 2.0 부츠 코어 블랙 클라우드 화이트" ]

if __name__ == "__main__":
    # 전체 결과를 저장할 리스트 초기화
    all_results = []

    for search_query in search_queries:
        # 검색어로 동영상 리스트 가져오기
        videos = youtube_search(api_key, search_query, max_results=100)

        # 검색 결과를 저장할 리스트 초기화
        results = []

        for video in videos:
            try:
                video_id = video['id']
                title = video['title']

                # 동영상 세부 정보 가져오기
                details = get_video_details(api_key, video_id)
                if 'error' in details:
                    logger.error(f"Skipping video ID {video_id} due to error in getting video details.")
                    continue
                description = details['description']
                view_count = details['view_count']
                like_count = details['like_count']
                published_at = details['published_at']
                channel_id = details['channel_id']

                # 댓글 가져오기
                comments = get_video_comments(api_key, video_id)
                comment_count = len(comments)

                # 각 댓글을 하나의 문자열로 결합
                comments_combined = " | ".join(comments)

                # 채널 구독자 수 가져오기
                subscriber_count = get_channel_subscriber_count(api_key, channel_id)

                # 결과 리스트에 추가
                results.append({
                    "검색어": search_query,
                    "영상ID": video_id,
                    "제목": title,
                    "게시글": description,
                    "조회수": view_count,
                    "좋아요수": like_count,
                    "게시일": published_at,
                    "댓글수": comment_count,
                    "각댓글내용": comments_combined,
                    "구독자수": subscriber_count
                })

            except HttpError as e:
                print(f"An HTTP error {e.resp.status} occurred for video ID {video['id']}: {e.content}")
                logger.error(f"An HTTP error {e.resp.status} occurred for video ID {video['id']}: {e.content}")
                continue
            except Exception as e:
                print(f"An unexpected error occurred for video ID {video['id']}: {str(e)}")
                logger.error(f"An unexpected error occurred for video ID {video['id']}: {str(e)}")
                continue

        # 각 검색어에 대한 결과를 전체 결과 리스트에 추가
        all_results.extend(results)

    # 데이터프레임으로 변환
    df = pd.DataFrame(all_results)

    # CSV 파일로 저장
    df.to_csv('youtube_data.csv', index=False, encoding='utf-8-sig')
    print("Data saved to youtube_data.csv")
