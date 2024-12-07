import torch

class DataLoader:
    def __init__(self, tokens, batch_size, context_length) -> None:
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_length = context_length
        self.current_position = 0

    def get_batch(self) -> torch.tensor:
        b, c = self.batch_size, self.context_length

        start_pos = self.current_position
        end_pos = self.current_position + b * c + 1

        add_data = -1  # Amount of additional data to fetch if we exceed token length
        if end_pos > len(self.tokens):
            add_data = end_pos - len(self.tokens)
            end_pos = len(self.tokens)

        d = self.tokens[start_pos:end_pos]
        if add_data != -1:
            d = torch.cat([d, self.tokens[:add_data]])

        x = (d[:-1]).view(b, c)  # Inputs
        y = (d[1:]).view(b, c)  # Targets

        self.current_position += b * c
        if self.current_position > len(self.tokens) - 1:
            self.current_position = 0
        return x, y

    def __iter__(self):
        """
        Make DataLoader iterable for use in for-loops.
        """
        self.current_position = 0  # Reset position at the start of iteration
        return self

    def __next__(self):
        """
        Return the next batch.
        """
        if self.current_position + self.batch_size * self.context_length >= len(self.tokens):
            raise StopIteration
        return self.get_batch()
