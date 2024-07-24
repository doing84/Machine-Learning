from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from bs4 import BeautifulSoup
import logging
import time
from queue import Queue

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('youtube_crawler.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

# 드라이버 풀 설정
class WebDriverPool:
    def __init__(self, size):
        self.pool = Queue(maxsize=size)
        for _ in range(size):
            self.pool.put(setup_driver())

    def get_driver(self):
        return self.pool.get()

    def return_driver(self, driver):
        self.pool.put(driver)

# 드라이버 풀 초기화
driver_pool = WebDriverPool(size=4)

def get_page_source(driver, url):
    driver.get(url)
    time.sleep(1.5)
    return driver.page_source

def parse_video_elements(page_source):
    soup = BeautifulSoup(page_source, 'html.parser')
    videos = []
    video_elements = soup.select('a#video-title')
    for video in video_elements:
        href = video.get('href')
        title = video.get('title')
        if href:
            video_id = href.split("v=")[-1]
            videos.append({'id': video_id, 'title': title})
    return videos

def youtube_search(query, published_after=None, published_before=None):
    youtube_search_url = f"https://www.youtube.com/results?search_query={query}"
    driver = driver_pool.get_driver()
    page_source = get_page_source(driver, youtube_search_url)

    videos = []
    last_height = driver.execute_script("return document.documentElement.scrollHeight")

    while True:
        new_videos = parse_video_elements(page_source)
        if len(new_videos) > 0:
            videos.extend(new_videos)
        
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(1.5)
        page_source = get_page_source(driver, youtube_search_url)

        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    driver_pool.return_driver(driver)  # 드라이버 반환
    logger.info(f"Total videos fetched: {len(videos)}")
    return videos

def parse_count(count_str):
    """
    Parse a count string like "1.1천", "2.5만" to an integer.
    """
    multipliers = {"천": 1000, "만": 10000}
    if count_str[-1] in multipliers:
        return int(float(count_str[:-1]) * multipliers[count_str[-1]])
    else:
        return int(count_str.replace(',', '').replace('명', ''))

def get_video_details(video_id):
    driver = driver_pool.get_driver()
    try:
        driver.get(f"https://www.youtube.com/watch?v={video_id}")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1.title")))

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        title = soup.select_one("h1.title").text if soup.select_one("h1.title") else ""

        description_element = soup.select_one("div#description")
        description = description_element.text if description_element else ""

        view_count_element = soup.select_one("span.view-count")
        view_count = ''.join(filter(str.isdigit, view_count_element.text)) if view_count_element else "0"

        like_count_element = soup.select_one("div#top-level-buttons-computed yt-formatted-string")
        if like_count_element:
            like_count_text = like_count_element.text
            try:
                like_count = parse_count(like_count_text)
            except ValueError:
                like_count = 0
        else:
            like_count = 0

        published_at = soup.select_one("div#info-strings yt-formatted-string").text if soup.select_one("div#info-strings yt-formatted-string") else ""
        channel_link_element = soup.select_one("a.yt-simple-endpoint.style-scope.yt-formatted-string")
        channel_id = ""
        if channel_link_element:
            channel_url = channel_link_element.get('href')
            if "/c/" in channel_url:
                channel_id = channel_url.split("/c/")[-1]
            elif "/channel/" in channel_url:
                channel_id = channel_url.split("/channel/")[-1]
            elif "/user/" in channel_url:
                channel_id = channel_url.split("/user/")[-1]

        return {
            'title': title,
            'description': description,
            'view_count': view_count,
            'like_count': like_count,
            'published_at': published_at,
            'channel_id': channel_id
        }
    except Exception as e:
        logger.error(f"Error fetching video details for ID {video_id}: {e}")
        return {}
    finally:
        driver_pool.return_driver(driver)

def get_video_comments(video_id):
    driver = driver_pool.get_driver()
    try:
        driver.get(f"https://www.youtube.com/watch?v={video_id}")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-comment-thread-renderer")))

        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        comments = []
        scroll_pause_time = 2  # 스크롤 후 대기 시간 설정

        while True:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(scroll_pause_time)

            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            comment_elements = soup.select("ytd-comment-thread-renderer #content-text")
            comments.extend([elem.text for elem in comment_elements])

        return comments
    except Exception as e:
        logger.error(f"Error fetching comments for video ID {video_id}: {e}")
        return []
    finally:
        driver_pool.return_driver(driver)

def get_channel_subscriber_count(channel_id):
    driver = driver_pool.get_driver()
    try:
        driver.get(f"https://www.youtube.com/{channel_id}")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "yt-formatted-string#subscriber-count")))

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        subscriber_count_element = soup.select_one("yt-formatted-string#subscriber-count")
        if subscriber_count_element:
            subscriber_count_text = subscriber_count_element.text
            try:
                subscriber_count = parse_count(subscriber_count_text)
            except ValueError:
                subscriber_count = 0
        else:
            subscriber_count = 0

        return subscriber_count
    except Exception as e:
        logger.error(f"Error fetching subscriber count for channel ID {channel_id}: {e}")
        return 0
    finally:
        driver_pool.return_driver(driver)
