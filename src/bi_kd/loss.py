import torch
import torch.nn.functional as F

def dot_product_scores(q_vectors, ctx_vectors):
    """
    calculates q-ctx dot product scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r

def cosine_scores(q_vectors, ctx_vectors):
    """
    calculates q-ctx cosine scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    r = F.cosine_similarity(q_vectors, ctx_vectors, dim=0)
    return r

class BiEncoderNllLoss(object):
    def __init__(self,
                 score_type="dot"):
        self.score_type = score_type
        
    def calc(
        self,
        q_vectors,
        ctx_vectors):
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)
            positive_idx_per_question = [i for i in range(q_num)]

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()
        return loss, correct_predictions_count

    def get_scores(self, q_vector, ctx_vectors):
        f = self.get_similarity_function()
        return f(q_vector, ctx_vectors)

    def get_similarity_function(self):
        if self.score_type == "dot":
            return dot_product_scores
        else:
            return cosine_scores
    
class BiEncoderDoubleNllLoss(object):
    def __init__(self,
                 score_type="dot", 
                 alpha = 0.5):
        self.score_type = score_type
        self.alpha = alpha
        
    def calc(
        self,
        q_vectors,
        ctx_vectors):
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            ctx_num = ctx_vectors.size(0)
            no_hard = int(ctx_num/q_num - 1)
            scores = scores.view(q_num, -1)
            
            positive_idx_per_question = [i for i in range(q_num)]
            scores2 = torch.randn(q_num, ctx_num - no_hard).to("cuda")
            for i in range(q_num):
                hard_neg_idx = [x for x in range((q_num+i* no_hard),(q_num+i* no_hard+ no_hard))]
                random_neg = [x for x in range(ctx_num) if x not in hard_neg_idx]
                subscores = self.get_scores(q_vectors[i], ctx_vectors[random_neg])
                subscores = subscores.view(1,-1)
                scores2[i] = subscores
                
        softmax_scores = F.log_softmax(scores, dim=1)
        softmax_scores2 = F.log_softmax(scores2, dim=1)

        loss1 = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        loss2 = F.nll_loss(
            softmax_scores2,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()
        return loss, correct_predictions_count

    def get_scores(self, q_vector, ctx_vectors):
        f = self.get_similarity_function()
        return f(q_vector, ctx_vectors)

    def get_similarity_function(self):
        if self.score_type == "dot":
            return dot_product_scores
        else:
            return cosine_scores