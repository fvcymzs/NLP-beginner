from typing import Tuple
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class MLP(nn.Module):
    # loss function is written into the model
    def __init__(self, in_features: int, hidden_features: int, class_num: int):
        """
        :param in_features: dimension of the input data
        :param hidden_features: dimension of the hidden layer
        :param class_num: number of class
        """
        super().__init__()
        self.full_conn1 = nn.Linear(in_features, hidden_features)
        self.tanh = nn.Tanh()
        self.full_conn2 = nn.Linear(hidden_features, class_num)
        self.loss_func = nn.CrossEntropyLoss()

    def get_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        get scores, for accuracy verification
        :param x: shape
        :return: scores
        """
        return self.full_conn2(self.tanh(self.full_conn1(x)))


class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """
        :param input_size: input size
        :param hidden_size: hidden size, can be changed by *2 or /2
        """
        super().__init__()
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)

    def forward(self, tensor_sentences: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        :param tensor_sentences: the sentences
        :param seq_lengths: actual length of sentences
        :return: result
        """
        packed_sentences = pack_padded_sequence(tensor_sentences, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.bi_lstm(packed_sentences)
        result, _ = pad_packed_sequence(output, batch_first=True)
        return result


class LocalInferenceModeling(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=2)

    def attention(self, x1_bar: torch.Tensor, seq_lengths1: torch.Tensor, x2_bar: torch.Tensor, seq_lengths2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        implement attention using dot product model, input a_bar and b_bar sentences，each word has same dimension，e_ij = b_bar_j.T @ a_bar_i = a_bar_i.T @ b_bar_j
        a_bar is the query vector, and e_ij is the degree to which the jth input vector is concerned when a_bar_i is given a task-related query, a_tilde_i = Σ softmax(e_ij) * b_bar_j
        b_bar is the query vector, and e_ij is the degree to which the jth input vector is concerned when b_bar_i is given a task-related query, b_tilde_j = Σ softmax(e_ij) * a_bar_i
        :param x1_bar: Corresponding to a_bar above, hidden_size see ESIM initialization input
        :param x2_bar: Corresponding to b_bar above, hidden_size see ESIM initialization input
        :param seq_lengths1: actual length of sentences
        :param seq_lengths2: actual length of sentences
        """
        # torch.bmm is considered batch_size group matrix multiplication
        e = torch.bmm(x1_bar, x2_bar.transpose(1, 2))

        batch_size, max_sentence_length1, max_sentence_length2 = e.shape

        # The actual length of each sentence is different. We shield the corresponding part of the pad and let the attention be on the pad part which is 0
        # seq_lengths1.unsqueeze(-1) generates shape: (batch_size, 1)
        # torch.arange(max_sentence_length1).expand(batch_size, -1) expands to generate shape: (batch_size, max_sentence_length1), each row is [0, 1, 2, ...]
        # torch.ge is responsible for comparison (greater or equal)
        # mask1 is torch.BoolTensor shape: (batch_size, max_sentence_length1)
        mask1 = torch.ge(torch.arange(max_sentence_length1, device=x1_bar.device).expand(batch_size, -1), seq_lengths1.unsqueeze(-1))
        # mask2 is torch.BoolTensor shape: (batch_size, max_sentence_length2)
        mask2 = torch.ge(torch.arange(max_sentence_length2, device=x1_bar.device).expand(batch_size, -1), seq_lengths2.unsqueeze(-1))

        # after masked_fill (using mask2) is applied to e, due to mask2.unsqueeze(1)
        # after softmax (softmax(e) still has the same shape as e), softmax(e) @ x2[k] is equivalent to softmax(e)[:, real_seq_length] @ [(x2[k]'s 0rd_word), (x2[k]'s 1st_word), (x2[k]'s 2nd_word)...].T
        # in this way, mask2 masks the word part of pad.
        softmax_e = self.softmax(e.masked_fill(mask2.unsqueeze(1), float('-inf')))
        x1_tilde = torch.bmm(softmax_e, x2_bar)
        softmax_e = self.softmax(e.transpose(1, 2).masked_fill(mask1.unsqueeze(1), float('-inf')))  # (batch, max_sentence_length2, max_sentence_length1)
        x2_tilde = torch.bmm(softmax_e, x1_bar)  # (batch, max_sentence_length2, hidden_size) 对应 b_tilde
        return x1_tilde, x2_tilde

    @staticmethod
    def enhancement(x_bar: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
        """
        :param x_bar: hidden_size see ESIM initialization input
        :param x_tilde: shape is same as x_bar
        """
        return torch.cat([x_bar, x_tilde, x_bar - x_tilde, x_bar * x_tilde], dim=-1)

    def forward(self, x1_bar: torch.Tensor, seq_lengths1: torch.Tensor, x2_bar: torch.Tensor, seq_lengths2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        for more details on parameters, see attention and enhancement
        :return: ma = [a_bar; a_tilde; a_bar − a_tilde; a_bar * a_tilde] mb = [b_bar; b_tilde; b_bar − b_tilde; b_bar * b_tilde]
        """
        x1_tilde, x2_tilde = self.attention(x1_bar, seq_lengths1, x2_bar, seq_lengths2)
        return self.enhancement(x1_bar, x1_tilde), self.enhancement(x2_bar, x2_tilde)


class InferenceComposition(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, class_num: int):
        """
        :param input_size: the output dimension of the previous layer; when actually assigning values, input_size = hidden_size * 4, hidden_size see ESIM initialization input
        :param hidden_size: hidden_size is the initialization input of ESIM
        :param class_num: number of class
        """
        super().__init__()
        # single-layer neural network (ReLU as activation function), mainly used to reduce model parameters to avoid overfitting
        self.F = nn.Linear(input_size, hidden_size)
        # use a 1-layer feedforward neural network with the ReLU activation
        self.relu = nn.ReLU()
        # (batch_size, max_seq_length_i, hidden_size) -> (batch_size, max_seq_length_i, hidden_size), output is the h_t of the last layer
        # (hidden_size // 2) for bidirectional
        self.BiLSTM = BiLSTM(input_size=hidden_size, hidden_size=hidden_size // 2)

        # (batch_size, max_seq_length_i, hidden_size) -> (batch_size, hidden_size)
        # v = [va_ave; va_max; vb_ave; vb_max]. (batch_size, 4 * hidden_size)
        self.MLP = MLP(in_features=4 * hidden_size, hidden_features=hidden_size, class_num=class_num)
        self.loss_func = self.MLP.loss_func

    def handle_x(self, m_x: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        :param m_x: equivalent to m_a/m_b above, hidden_size see ESIM initialization input
        :param seq_lengths: actual length of sentences
        :return: [va_ave; va_max] or [vb_ave; vb_max]
        """
        v_x_t = self.BiLSTM(self.relu(self.F(m_x)), seq_lengths)  # (batch_size, max_sentence_length, hidden_size)  hidden_size 见 ESIM 的初始化输入；对应论文的 x_a_t/v_b_t
        max_sentence_length = m_x.shape[1]
        # for over-time pooling
        v_x_t_transpose = v_x_t.transpose(1, 2)
        v_x_avg = F.avg_pool1d(v_x_t_transpose, kernel_size=max_sentence_length).squeeze(-1)
        v_x_max = F.max_pool1d(v_x_t_transpose, kernel_size=max_sentence_length).squeeze(-1)
        return torch.cat([v_x_avg, v_x_max], dim=1)

    def get_scores(self, m_x1: torch.Tensor, seq_lengths1: torch.Tensor, m_x2: torch.Tensor, seq_lengths2: torch.Tensor) -> torch.Tensor:
        """
        get the score for accuracy verification
        :param m_x1: Equivalent to m_a above, hidden_size See ESIM initialization input
        :param m_x2: Equivalent to m_b above, hidden_size See ESIM initialization input
        :param seq_lengths1: actual length of sentences
        :param seq_lengths2: actual length of sentences
        """
        # composition layer
        # v = [va_ave; va_max; vb_ave; vb_max]
        v = torch.cat([self.handle_x(m_x1, seq_lengths1), self.handle_x(m_x2, seq_lengths2)], dim=-1)
        return self.MLP.get_scores(v)


class ESIM(nn.Module):
    def __init__(self, embedding: nn.Embedding, embedding_size: int, hidden_size: int, class_num: int):
        """
        :param embedding: pre-trained word vectors
        :param embedding_size: word vector dimension
        :param hidden_size: due to the existence of bidirectional, will be changed by / 2
        :param class_num: number of class
        """
        super().__init__()
        assert hidden_size % 2 == 0
        # (batch_size, max_seq_length_i) -> (batch_size, max_seq_length_i, embedding_size)
        self.embedding = embedding
        # input encoding
        self.BiLSTM = BiLSTM(input_size=embedding_size, hidden_size=hidden_size // 2)
        # local inference modeling
        # (batch_size, max_seq_length_i, hidden_size)  -> (batch_size, max_seq_length_i, hidden_size*4)
        self.local_inference_modeling = LocalInferenceModeling()
        # inference composition
        # (batch_size, max_seq_length_i, hidden_size * 4) -> loss
        self.inference_composition = InferenceComposition(input_size=hidden_size*4, hidden_size=hidden_size, class_num=class_num)
        self.loss_func = self.inference_composition.loss_func

    def get_scores(self, x1_indices: torch.Tensor, seq_lengths1: torch.Tensor, x2_indices: torch.Tensor, seq_lengths2: torch.Tensor) -> torch.Tensor:
        """
        get the score for accuracy verification
        :param x1_indices: premise sentence group, index of vocab corresponding to each word
        :param x2_indices: premise sentence group, index of vocab corresponding to each word
        :param seq_lengths1: actual length of sentences
        :param seq_lengths2: actual length of sentences
        :return: scores
        """
        x1 = self.embedding(x1_indices)
        x2 = self.embedding(x2_indices)
        # input encoding
        x1_bar = self.BiLSTM(x1, seq_lengths1)
        x2_bar = self.BiLSTM(x2, seq_lengths2)
        # local inference modeling
        m_x1, m_x2 = self.local_inference_modeling.forward(x1_bar, seq_lengths1, x2_bar, seq_lengths2)
        # inference composition
        scores = self.inference_composition.get_scores(m_x1, seq_lengths1, m_x2, seq_lengths2)
        return scores

    def forward(self, x1_indices: torch.Tensor, seq_lengths1: torch.Tensor, x2_indices: torch.Tensor, seq_lengths2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        get loss for gradient descent
        :param x1_indices: premise sentence group, index of vocab corresponding to each word
        :param x2_indices: premise sentence group, index of vocab corresponding to each word
        :param seq_lengths1: actual length of sentences
        :param seq_lengths2: actual length of sentences
        :param y: y[i] is the true classification value of x[i], and 0 <= y[i] < CLASS_NUM
        """
        scores = self.get_scores(x1_indices, seq_lengths1, x2_indices, seq_lengths2)  # scores shape: (batch_size, class_num)
        return self.loss_func(scores, y)
