import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

dataset = load_dataset("yelp_review_full")
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(1000))
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")
tokenized_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from peft import LoraConfig, get_peft_model, TaskType

# Define LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS, # this is necessary
    inference_mode=True
)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # see % trainable parameters

training_args = TrainingArguments(output_dir="bert_peft_trainer")
bert_peft_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset, # training dataset requires column input_ids
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
)
bert_peft_trainer.train()
bert_peft_trainer.model.save_pretrained("bert-peft")
                                        
# from peft import PeftModel
# original_model = AutoModelForSequenceClassification.from_pretrained(
#   model.config["_name_or_path"]
# )
# original_with_adapter = PeftModel.from_pretrained(
#   original_model, "bert-peft" # bert-peft; the folder of the saved adapter
# )
# merged_model = original_with_adapter.merge_and_unload()
# merged_model.save_pretrained("merged-model")