import pandas as pd
from youtube_analyze_sentiment_naver import combine_data_files, extract_aspect_sentiments, split_review_sentiments, check_and_remove_duplicates

def main():

    try:
        data_path = "./data"

        # 데이터 파일 하나로 합치기
        data = combine_data_files(data_path)
        df = pd.DataFrame(data)  
        print(df)
        df.to_csv("combine_data_files.csv", index=False, encoding='utf-8-sig')

        # 데이터프레임에 각 리뷰의 감정 점수를 추가
        df['AspectSentiments'] = df['Aspects'].apply(extract_aspect_sentiments)
        print(df)
        df.to_csv("extract_aspect_sentiments.csv", index=False, encoding='utf-8-sig')

        # 각 리뷰별 감정 점수 분리    
        sentiment_df = split_review_sentiments(df)
        print(sentiment_df)
        sentiment_df.to_csv("split_review_sentiments.csv", index=False, encoding='utf-8-sig')

        # 중복된 리뷰 확인 및 제거
        sentiment_df = check_and_remove_duplicates(sentiment_df)
        print(sentiment_df)
        sentiment_df.to_csv("check_and_remove_duplicates.csv", index=False, encoding='utf-8-sig')

        # CSV 파일로 저장
        sentiment_df.to_csv("text_sentiment.csv", index=False, encoding='utf-8-sig')
        print("Has been saved to text_sentiment.csv")

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except pd.errors.EmptyDataError as e:
        print(f"EmptyDataError: {e}")
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()     