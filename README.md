# bip-talent-week-2025

Project developed for the 24-hour [BIP AI Talent Week 2025 Challenge](https://www.vgen.it/it/bip-ai-talent-week-2025-partecipazione/) and awarded with the 1st place.


## Summary

Implementation of a ML pipeline for the binary classification of Hotel review as "positive" or "negative".

The solution developed employes a [pretrained](https://huggingface.co/gosorio/robertaSentimentFT_TripAdvisor) version of the [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base) encoder architecture, replaces the classification head and fine-tunes it on the provided train dataset.

The trained model is available at the following [HuggingFace repo](https://huggingface.co/Matteo-Pannacci/bip-talent-week-2025).
