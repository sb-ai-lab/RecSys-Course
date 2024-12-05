from typing import Dict, List
import torch

from replay.data.nn import (
    SequenceTokenizer,
)
from replay.models.nn.sequential import SasRec
from replay.models.nn.sequential.sasrec import (
    SasRecPredictionBatch,
)

from mlflow.pyfunc import PythonModel, PythonModelContext


class Model(PythonModel):
    def __init__(self, model: SasRec, tokenizer: SequenceTokenizer):
        self.model = model.to("cpu").eval()
        self.max_seq_len = self.model._model.max_len

        self.item_mapping = tokenizer.item_id_encoder.mapping["item_id"]
        self.item_inverse_mapping = tokenizer.item_id_encoder.inverse_mapping["item_id"]

    def predict(self, _: PythonModelContext, data: List[int]) -> List[int]:
        encoded_seq = self._encode_data(data)

        item_sequence = torch.Tensor(encoded_seq).unsqueeze(0)[:, -int(self.max_seq_len):]
        padding_mask = torch.ones_like(item_sequence, dtype=torch.bool)

        batch = SasRecPredictionBatch(
            query_id=torch.arange(0, item_sequence.shape[0], 1).long(),
            padding_mask=padding_mask.bool(),
            features={"item_id_seq": item_sequence.long()},
        )
        with torch.no_grad():
            scores = self.model.predict_step(batch, 0)
        sorted_scores = torch.topk(scores, k=10).indices[0].tolist()
        return self._decode_data(sorted_scores)

    def _encode_data(self, data: List[int]) -> List[int]:
        encoded_data = [self.item_mapping[item_id] for item_id in data if item_id in self.item_mapping]
        return encoded_data[-self.max_seq_len :]

    def _decode_data(self, data: List[int]) -> List[int]:
        return [self.item_inverse_mapping[ix] for ix in data]
