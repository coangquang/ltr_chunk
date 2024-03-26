import torch
import torch.nn.functional as F

class CrossEncoderNllLoss(object):
    def __init__(self,
                 score_type="dot"):
        self.score_type = score_type
        
    def calc(
        self,
        logits,
        labels):
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Return: a tuple of loss value and amount of correct predictions per batch
        """

        #if len(q_vectors.size()) > 1:
        #    q_num = q_vectors.size(0)
        #    scores = scores.view(q_num, -1)
        #    positive_idx_per_question = [i for i in range(q_num)]

        softmax_scores = F.log_softmax(logits, dim=1)
        print("softmax", softmax_scores)
        loss = F.nll_loss(
            softmax_scores,
            labels,
            reduction="mean",
        )

        #max_score, max_idxs = torch.max(softmax_scores, 1)
        #correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()
        return loss#, correct_predictions_count

