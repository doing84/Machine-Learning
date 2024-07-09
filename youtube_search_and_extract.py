from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging

# 로깅 설정
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('youtube_api.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# 동영상 리스트 가져오기
def youtube_search(api_key, query, max_results=50):
    youtube = build('youtube', 'v3', developerKey=api_key)
    videos = []
    next_page_token = None

    while len(videos) < max_results:
        try:
            search_response = youtube.search().list(
                q=query,
                part='id,snippet',
                maxResults=min(max_results - len(videos), 50),
                pageToken=next_page_token  # 다음 페이지 토큰 설정
            ).execute()

            for search_result in search_response.get('items', []):
                if search_result['id']['kind'] == 'youtube#video':
                    video_id = search_result['id']['videoId']
                    video_title = search_result['snippet']['title']
                    videos.append({'id': video_id, 'title': video_title})

            next_page_token = search_response.get('nextPageToken')  # 다음 페이지 토큰 추출
            if not next_page_token:
                break  # 다음 페이지 토큰이 없으면 반복 종료

        except HttpError as e:
            logger.error(f'An HTTP error {e.resp.status} occurred: {e.content}')
            break

    return videos


def get_video_details(api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    try:
        video_response = youtube.videos().list(
            id=video_id,
            part='statistics,snippet'
        ).execute()

        for video_result in video_response.get('items', []):
            title = video_result['snippet'].get('title', 'N/A')
            description = video_result['snippet'].get('description', 'N/A')
            view_count = video_result['statistics'].get('viewCount', 'N/A')
            like_count = video_result['statistics'].get('likeCount', 'N/A')
            published_at = video_result['snippet'].get('publishedAt', 'N/A')
            channel_id = video_result['snippet'].get('channelId', 'N/A')
            return {
                'title': title,
                'description': description,
                'view_count': view_count,
                'like_count': like_count,
                'published_at': published_at,
                'channel_id': channel_id
            }

    except HttpError as e:
        logger.error(f'An HTTP error {e.resp.status} occurred: {e.content}')
        return {
            'error': True,
            'message': str(e)
        }

def get_video_comments(api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    try:
        comment_response = youtube.commentThreads().list(
            videoId=video_id,
            part='snippet',
            maxResults=100
        ).execute()

        while comment_response:
            for comment_result in comment_response.get('items', []):
                comment = comment_result['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            if 'nextPageToken' in comment_response:
                comment_response = youtube.commentThreads().list(
                    videoId=video_id,
                    part='snippet',
                    maxResults=100,
                    pageToken=comment_response['nextPageToken']
                ).execute()
            else:
                break

    except HttpError as e:
        logger.error(f'An HTTP error {e.resp.status} occurred: {e.content}')
        if 'commentsDisabled' in str(e.content):
            print(f"Comments are disabled for video ID {video_id}.")
            logging.warning(f"Comments are disabled for video ID {video_id}.")
        return []

    return comments

def get_channel_subscriber_count(api_key, channel_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    try:
        channel_response = youtube.channels().list(
            id=channel_id,
            part='statistics'
        ).execute()

        for channel_result in channel_response.get('items', []):
            subscriber_count = channel_result['statistics'].get('subscriberCount', 'N/A')
            return subscriber_count

    except HttpError as e:
        logger.error(f'An HTTP error {e.resp.status} occurred: {e.content}')
        return 'N/A'
