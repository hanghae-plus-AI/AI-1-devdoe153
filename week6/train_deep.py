import os
import sys
import json
import logging
import wandb
from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import train_test_split

import datasets
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed
)

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default="gpt2")
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})
    block_size: int = field(default=1024)
    num_workers: Optional[int] = field(default=None)

@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./output", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    run_name: Optional[str] = field(default="gpt-finetuning-run")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((Arguments, CustomTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    # 시드 설정
    set_seed(training_args.seed)

    # wandb 초기화
    wandb.init(project="gpt-finetuning", name=training_args.run_name)

    # corpus.json 파일 불러오기
    try:
        with open('corpus.json', 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        if isinstance(corpus, dict) and 'data' in corpus:
            corpus = corpus['data']
        if not isinstance(corpus, list):
            raise ValueError("Corpus should be a list of texts or a dictionary with a 'data' key containing a list of texts.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        logger.error("Please check your corpus.json file for correct JSON format.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading corpus.json: {e}")
        sys.exit(1)

    # 데이터 구조 처리
    if 'data' in corpus:
        # instruction과 response를 결합하여 학습 데이터 생성
        texts = []
        for item in corpus['data']:
            combined_text = f"Instruction: {item['instruction']}\nResponse: {item['response']}"
            texts.append(combined_text)
    else:
        raise ValueError("올바른 데이터 형식이 아닙니다. 'data' 키가 필요합니다.")

    # 데이터가 충분한지 확인
    if len(texts) < 2:
        raise ValueError("데이터가 너무 적습니다. 최소 2개 이상의 데이터가 필요합니다.")

    logger.info(f"총 {len(texts)}개의 데이터를 로드했습니다.")

    # 8:2 비율로 train과 validation data 나누기
    train_data, valid_data = train_test_split(texts, test_size=0.2, random_state=42)

    # 데이터셋 생성
    train_dataset = Dataset.from_dict({"text": train_data})
    valid_dataset = Dataset.from_dict({"text": valid_data})

    # 토크나이저 및 모델 로드
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=args.torch_dtype
    )

    # 토크나이저 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.block_size)

    # 데이터 전처리
    tokenized_train = train_dataset.map(tokenize_function, batched=True, num_proc=args.num_workers, remove_columns=["text"])
    tokenized_valid = valid_dataset.map(tokenize_function, batched=True, num_proc=args.num_workers, remove_columns=["text"])

    # Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=default_data_collator,
    )

    # 학습 시작
    try:
        train_result = trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)

    # 모델 저장
    trainer.save_model()

    # 메트릭 로깅
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 검증 데이터에 대한 평가
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

if __name__ == "__main__":
    main()
