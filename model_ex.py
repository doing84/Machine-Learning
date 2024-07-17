import os
import shutil
import pandas as pd
from model_train import train_prepare_data, create_dataset, final_evaluation, get_best_checkpoint, cross_validate
from transformers import AutoTokenizer

def main():
    # 기존 체크포인트 로그 파일과 폴더 삭제
    if os.path.exists('checkpoint_log.txt'):
        os.remove('checkpoint_log.txt')
    shutil.rmtree('./results', ignore_errors=True)
    shutil.rmtree('./best_checkpoint', ignore_errors=True)

    file_path = "text_sentiment.csv"
    model_name = "beomi/KcELECTRA-base-v2022"

    # 데이터 준비
    train_text, test_text, train_label, test_label, label_encoder = train_prepare_data(file_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 데이터프레임 생성
    df = pd.DataFrame({'Text': train_text.tolist() + test_text.tolist(), 'Sentiment': train_label.tolist() + test_label.tolist()})

    # 교차 검증 수행
    accuracies = cross_validate(df, tokenizer, model_name, label_encoder, n_splits=5)
    print("Cross-validation accuracies:", accuracies)
    print("Average accuracy:", sum(accuracies) / len(accuracies))

    # 최적의 체크포인트 찾기
    best_checkpoint = get_best_checkpoint()
    print("Best checkpoint:", best_checkpoint)

    # 최적의 체크포인트 절대 경로로 변환
    if best_checkpoint:
        best_checkpoint = os.path.abspath(best_checkpoint).replace('\\', '/')
        print("Absolute path for the best checkpoint:", best_checkpoint)

        # 필요한 파일이 체크포인트에 있는지 확인
        required_files = ['pytorch_model.bin', 'config.json', 'training_args.bin']
        files_exist = all(os.path.exists(os.path.join(best_checkpoint, f)) for f in required_files)

        if files_exist:
            print(f"All required files exist in the checkpoint path: {best_checkpoint}")
            final_accuracy = final_evaluation(train_text, train_label, test_text, test_label, tokenizer, best_checkpoint, label_encoder)
            print("Final accuracy with the best checkpoint:", final_accuracy)
        else:
            print(f"Checkpoint path {best_checkpoint} does not contain all required files: {required_files}")
            raise FileNotFoundError(f"Checkpoint path {best_checkpoint} does not contain all required files: {required_files}")
    else:
        print("No best checkpoint found.")

if __name__ == "__main__":
    main()

