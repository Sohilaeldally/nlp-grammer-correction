   def __getitem__(self, idx):
        input_text = self.texts[idx]
        target_text = self.labels[idx]
        
        encoding = self.tokenizer(
            input_text,
            text_target=target_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        return item