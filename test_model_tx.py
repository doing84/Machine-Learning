import os
import pandas as pd
from test_model_train import train_prepare_data, create_dataset, final_evaluation, get_best_checkpoint
from transformers import AutoTokenizer

if __name__ == "__main__":
    file_path = "text_sentiment.csv"
    model_name = "beomi/KcELECTRA-base-v2022"

    train_text, test_text, train_label, test_label, label_encoder = train_prepare_data(file_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    best_checkpoint = get_best_checkpoint()
    print("Best checkpoint:", best_checkpoint)

    best_checkpoint = os.path.abspath(best_checkpoint).replace('\\', '/')
    print("Absolute path for the best checkpoint:", best_checkpoint)

    required_files = ['pytorch_model.bin', 'config.json', 'training_args.bin']
    files_exist = all(os.path.exists(os.path.join(best_checkpoint, f)) for f in required_files)

    if files_exist:
        print(f"All required files exist in the checkpoint path: {best_checkpoint}")
        final_accuracy = final_evaluation(train_text, train_label, test_text, test_label, tokenizer, best_checkpoint, label_encoder)
        print("Final accuracy with the best checkpoint:", final_accuracy)
    else:
        print(f"Checkpoint path {best_checkpoint} does not contain all required files: {required_files}")
        # 필요한 파일이 없을 경우 처리 (예: 예외 던지기)
        raise FileNotFoundError(f"Checkpoint path {best_checkpoint} does not contain all required files: {required_files}")
