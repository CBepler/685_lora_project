#!/bin/bash
# Extract all metrics from all training logs

echo "Extracting comprehensive metrics from all logs..."
echo ""

for model_dataset in "lora_imdb_40293858" "sparse_lora_imdb_40294631" "qlora_imdb_40296840" "hira_imdb_40293867" "lora_sst2_40293857" "sparse_lora_sst2_40294630" "qlora_sst2_40296839" "hira_sst2_40293866" "lora_wikitext2_40294194" "sparse_lora_wikitext2_40294633" "qlora_wikitext2_40296841" "hira_wikitext2_40294197"; do

    log_file="logs/${model_dataset}.out"

    if [ -f "$log_file" ]; then
        echo "=== $model_dataset ==="

        # Training time
        grep "Total training time:" "$log_file" 2>/dev/null

        # Accuracy
        grep -E "Test Accuracy|Validation Accuracy:" "$log_file" 2>/dev/null | tail -1

        # F1
        grep -E "Test F1|Validation F1:" "$log_file" 2>/dev/null | tail -1

        # Perplexity / Loss
        grep -E "Test Perplexity|Validation Loss:|Training Loss:" "$log_file" 2>/dev/null | tail -3

        # Throughput
        grep -E "Throughput|samples/second" "$log_file" 2>/dev/null | head -2

        # Memory
        grep -E "Peak GPU|GPU allocated|Memory footprint|Model memory" "$log_file" 2>/dev/null | head -5

        echo ""
    fi
done
