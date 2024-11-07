from datasets import load_dataset
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer

def main():
    # 모델과 토크나이저 로드
    model_id = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # 데이터셋 로드
    dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k")

    # 토크나이저 설정
    tokenizer.pad_token = tokenizer.eos_token
    max_length = 512

    # 데이터 전처리 함수
    def preprocess_function(examples):
        prompts = [f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                   for instruction, output in zip(examples['instruction'], examples['output'])]
        return tokenizer(prompts, truncation=True, max_length=max_length, padding="max_length")

    # 데이터셋 전처리
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    # LoRA r 값 목록
    lora_r_values = [8, 128, 256]

    # 각 LoRA r 값에 대해 학습 수행
    for r in lora_r_values:
        print(f"Training with lora_r = {r}")

        # wandb 설정
        wandb.init(project="Hanghae99", name=f"lora_r_{r}")

        # LoRA 설정
        lora_config = LoraConfig(
            r=r,
            target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )

        # 모델에 LoRA 적용
        lora_model = get_peft_model(model, lora_config)

        # 학습 인자 설정
        training_args = TrainingArguments(
            output_dir=f"./opt-code-alpaca-lora-r{r}",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            warmup_steps=10,
            max_steps=100,
            logging_steps=10,
            fp16=True,
            save_total_limit=3,
            report_to="wandb"
        )

        # 데이터 콜레이터 설정
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Trainer 초기화 및 학습
        trainer = SFTTrainer(
            model=lora_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator,
        )

        print('학습전 Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')

        # 학습 시작
        trainer.train()

        # 모델 저장
        trainer.save_model(f"./opt-code-alpaca-lora-r{r}-final")
        tokenizer.save_pretrained(f"./opt-code-alpaca-lora-r{r}-final")

        print('학습 후 Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')

        # wandb 종료
        wandb.finish()

        # GPU 캐시 정리
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
