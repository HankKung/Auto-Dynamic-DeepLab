import numpy as np
import torch
import torch.nn.functional as F
from genotypes import PRIMITIVES
from genotypes import Genotype

def network_layer_to_space(net_arch):
    for i, layer in enumerate(net_arch):
        if i == 0:
            space = np.zeros((1, 4, 3))
            space[0][layer][0] = 1
            prev = layer
        else:
            if layer == prev + 1:
                sample = 0
            elif layer == prev:
                sample = 1
            elif layer == prev - 1:
                sample = 2
            space1 = np.zeros((1, 4, 3))
            space1[0][layer][sample] = 1
            space = np.concatenate([space, space1], axis=0)
            prev = layer
    return space

def z(b):
    m = torch.mean(b)

    sd=0
    for i in b:
        sd+= (i-m)**2
    sd = sd / b.shape[0]
    sd = torch.sqrt(sd)

    for i in range(b.shape[0]):
        b[i] = (b[i]-m)/sd
    return b

def minmax(pred):
	minimum = min(pred)
	maximum = max(pred)
	m = maximum - minimum
	ret = torch.tensor (torch.randn(pred.shape[0]))
	for i in range(pred.shape[0]):
		ret[i] = (pred[i]-minimum)/m
	return ret

class Decoder(object):
    def __init__(self, alphas_d, alphas_c, betas, steps):
        self._betas = betas
        self._alphas_d = alphas_d
        self._alphas_c = alphas_c
        self._steps = steps
        self._num_layers = len(self._betas)
        self.network_space = torch.zeros(12, 4, 3)
        print(self._betas)
        for layer in range(len(self._betas)):
            if layer == 0:
                self._betas[layer][0][1:] = z(self._betas[layer][0][1:])
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1)
                print(self.network_space[layer][0][1:])
                print(F.softmax(self._betas[layer][0][1:], dim=-1))
            elif layer == 1:
                self._betas[layer][0][1:] = z(self._betas[layer][0][1:])
                self._betas[layer][1] = z(self._betas[layer][1])
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)

            elif layer == 2:
                self._betas[layer][0][1:] = z(self._betas[layer][0][1:])
                self._betas[layer][1] = z(self._betas[layer][1])
                self._betas[layer][2] = z (self._betas[layer][2])
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)
                self.network_space[layer][2] = F.softmax(self._betas[layer][2], dim=-1)
            else:
                self._betas[layer][0][1:] = z(self._betas[layer][0][1:])
                self._betas[layer][1] = z (self._betas[layer][1])
                self._betas[layer][2] = z(self._betas[layer][2])
                self._betas[layer][3][:2] = z(self._betas[layer][3][:2])
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)
                self.network_space[layer][2] = F.softmax(self._betas[layer][2], dim=-1)
                self.network_space[layer][3][:2] = F.softmax(self._betas[layer][3][:2], dim=-1)
        print(self.network_space)
    def viterbi_decode(self):
        prob_space = np.zeros((self.network_space.shape[:2]))
        path_space = np.zeros((self.network_space.shape[:2])).astype('int8')

        for layer in range(self.network_space.shape[0]):
            if layer == 0:
                prob_space[layer][0] = self.network_space[layer][0][1]
                prob_space[layer][1] = self.network_space[layer][0][2]
                path_space[layer][0] = 0
                path_space[layer][1] = -1
            else:
                for sample in range(self.network_space.shape[1]):
                    if layer - sample < - 1:
                        continue
                    local_prob = []
                    for rate in range(self.network_space.shape[2]):  # k[0 : ➚, 1: ➙, 2 : ➘]
                        if (sample == 0 and rate == 2) or (sample == 3 and rate == 0):
                            continue
                        else:
                            local_prob.append(prob_space[layer - 1][sample + 1 - rate] *
                                              self.network_space[layer][sample + 1 - rate][rate])
                    prob_space[layer][sample] = np.max(local_prob, axis=0)
                    rate = np.argmax(local_prob, axis=0)
                    path = 1 - rate if sample != 3 else -rate
                    path_space[layer][sample] = path  # path[1 : ➚, 0: ➙, -1 : ➘]

        output_sample = prob_space[-1, :].argmax(axis=-1)
        actual_path = np.zeros(12).astype('uint8')
        actual_path[-1] = output_sample
        for i in range(1, self._num_layers):
            actual_path[-i - 1] = actual_path[-i] + path_space[self._num_layers - i, actual_path[-i]]

        return actual_path, network_layer_to_space(actual_path)

    def genotype_decode(self):

        def _parse(alphas, steps):
            gene = []
            start = 0
            n = 2
            for i in range(steps):
                end = start + n
                edges = sorted(range(start, end), key=lambda x: -np.max(alphas[x, 4:]))  # ignore none value
                top2edges = edges[:2]
                for j in top2edges:
                    best_op_index = np.argmax(alphas[j][4:])+4  # this can include none op
                    gene.append([j, best_op_index])
                start = end
                n += 1
            return np.array(gene)

        normalized_alphas_d = F.softmax(self._alphas_d, dim=-1).data.cpu().numpy()
        normalized_alphas_c = F.softmax(self._alphas_c, dim=-1).data.cpu().numpy()

        gene_cell_d = _parse(normalized_alphas_d, 4)
        gene_cell_c = _parse(normalized_alphas_c, 5)

        return gene_cell_d, gene_cell_c