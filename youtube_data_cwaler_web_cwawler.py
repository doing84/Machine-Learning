import os
import sys
import pandas as pd
import logging
import gc
from datetime import datetime
from threading import Thread, Lock
from queue import Queue
from youtube_search_and_extract_web_crawler import youtube_search, get_video_details, get_video_comments, get_channel_subscriber_count

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('youtube_data_crawler.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# 검색어 리스트 설정
search_queries = ['비비안웨스트우드 실버']  # 다른 검색어를 추가할 수 있습니다.

# 날짜 범위 설정
published_after = '2021-04-12T00:00:00Z'
published_before = '2024-07-22T23:59:59Z'

# 락 객체 생성
lock = Lock()

def process_video(queue):
    while True:
        video_info = queue.get()
        if video_info is None:
            break
        video, search_query = video_info
        try:
            video_id = video['id']
            title = video['title']
            logger.info(f"Processing video ID: {video_id}, Title: {title}")

            # 동영상 세부 정보 가져오기
            details = get_video_details(video_id)
            if not details:
                logger.error(f"Skipping video ID {video_id} due to error in getting video details.")
                continue
            
            description = details.get('description', '')
            view_count = details.get('view_count', '0')
            like_count = details.get('like_count', '0')
            published_at = details.get('published_at', '')
            channel_id = details.get('channel_id', '')

            logger.info(f"Video details fetched: Views - {view_count}, Likes - {like_count}")

            # 댓글 가져오기
            comments = get_video_comments(video_id)
            comment_count = len(comments)

            logger.info(f"Fetched {comment_count} comments for video ID: {video_id}")

            # 각 댓글을 하나의 문자열로 결합
            comments_combined = " | ".join(comments)

            # 채널 구독자 수 가져오기
            subscriber_count = get_channel_subscriber_count(channel_id)

            logger.info(f"Channel ID: {channel_id}, Subscriber count: {subscriber_count}")

            # 결과 딕셔너리 반환
            result = {
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
            }

            logger.info(f"Appending result: {result}")

            # 락을 사용하여 공유 자원에 접근
            with lock:
                all_results.append(result)

        except Exception as e:
            logger.error(f"An unexpected error occurred for video ID {video_id}: {str(e)}")
        finally:
            queue.task_done()
            gc.collect()

if __name__ == "__main__":
    # 전체 결과를 저장할 리스트 초기화
    all_results = []
    video_queue = Queue()

    for search_query in search_queries:
        logger.info(f"Searching for query: {search_query}")

        # 검색어로 동영상 리스트 가져오기
        videos = youtube_search(search_query, published_after=published_after, published_before=published_before)
        logger.info(f"Fetched {len(videos)} videos.")
        
        # 검색 결과를 20개로 제한 (테스트 목적)
        videos = videos[:20]

        # 스레드 풀 초기화
        num_worker_threads = 6  # 단일 스레드 사용
        threads = []
        for i in range(num_worker_threads):
            t = Thread(target=process_video, args=(video_queue,))
            t.start()
            threads.append(t)

        for video in videos:
            video_queue.put((video, search_query))

        # 모든 작업이 완료되면 None을 큐에 추가하여 스레드를 종료
        video_queue.join()
        for i in range(num_worker_threads):
            video_queue.put(None)
        for t in threads:
            t.join()

    # 가비지 컬렉션 호출
    gc.collect()

    # 결과를 데이터프레임으로 저장
    logger.info(f"Total results collected: {len(all_results)}")
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv('youtube_data.csv', index=False, encoding='utf-8-sig')
        logger.info("Data saved to youtube_data.csv")
    else:
        logger.error("No results to save.")
