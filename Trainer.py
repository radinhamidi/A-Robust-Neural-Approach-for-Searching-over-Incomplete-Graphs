from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import eval_Z
from args import *
from torch_geometric.utils import structured_negative_sampling 
from torch_geometric.nn import Node2Vec

class Trainer(object):
    def __init__(self, model, X, edge_index, args, raw_embeddings=None, wl_embedding=None, int_embeddings=None, hop_embeddings=None):
        self.model = model
        self.lr = args.lr

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=2e-6)

        self.bestResult = -1
        self.beta = args.beta
        self.gamma = args.gamma
        self.sigma = args.sigma

        self.X = X
        self.raw_embeddings = raw_embeddings
        self.wl_embedding = wl_embedding
        self.int_embeddings = int_embeddings
        self.hop_embeddings = hop_embeddings
        self.edge_index = edge_index

        self.node_num, self.attr_num = X.shape

        self.sim_loss_m = args.m

        self.node2vec = Node2Vec(self.edge_index, embedding_dim=model.output_dim, walk_length=20,
                context_size=10, walks_per_node=10,
                num_negative_samples=1, p=1, q=1, sparse=True).to(self.X.device)

    def train_mini_batch(self):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def tf_loss(self):
        attr_emb = self.model.encoder(torch.eye(self.attr_num).to(self.X.device))
        return (attr_emb.norm(dim=1) * self.X.sum(dim=0)).mean()

    def ae_loss(self):
        return F.mse_loss(self.X, self.model.decoder(self.model.encoder(self.X)))

    # def bert_ae_loss(self):
    #     node_emb = self.model.bert_forward(self.X, self.edge_index, self.raw_embeddings, self.wl_embedding, self.int_embeddings, self.hop_embeddings)
    #     return F.mse_loss(self.X, self.model.decoder(node_emb))

    # def bert_tf_loss(self):
    #     attr_emb = self.model.encoder(torch.eye(self.attr_num).to(self.X.device))
    #     return (attr_emb.norm(dim=1) * self.X.sum(dim=0)).mean()

    def sim_loss(self):
        node_emb = self.model.forward(self.X, self.edge_index)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()

    def sage_sim_loss(self):
        node_emb = self.model.sage_forward(self.X, self.edge_index)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()


    def sage_decoder_sim_loss(self):
        node_emb = self.model.sage_decoder_forward(self.X, self.edge_index)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()


    def gcn_sim_loss(self):
        node_emb = self.model.gcn_forward(self.X, self.edge_index)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()

    def gcn_decoder_sim_loss(self):
        node_emb = self.model.gcn_decoder_forward(self.X, self.edge_index)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()


    def gat_sim_loss(self):
        node_emb = self.model.gat_forward(self.X, self.edge_index)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()


    def gat_decoder_sim_loss(self):
        node_emb = self.model.gat_decoder_forward(self.X, self.edge_index)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()

    def cheb_sim_loss(self):
        node_emb = self.model.cheb_forward(self.X, self.edge_index)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()


    def cheb_decoder_sim_loss(self):
        node_emb = self.model.cheb_decoder_forward(self.X, self.edge_index)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()


    def node2vec_sim_loss(self):
        node_emb = self.node2vec.forward(batch=None)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()


    def node2vec_decoder_sim_loss(self):
        node_emb = self.node2vec.forward(batch=None)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()


    def bert_sim_loss(self):
        node_emb = self.model.bert_forward(self.X, self.edge_index, self.raw_embeddings, self.wl_embedding, self.int_embeddings, self.hop_embeddings)

        src_nodes, pos_nodes, neg_nodes = structured_negative_sampling(self.edge_index)

        edges = torch.cat([torch.stack([src_nodes, pos_nodes]),torch.stack([src_nodes, neg_nodes])],dim=-1)

        # sim_label= torch.norm(self.dist_M[src_nodes]-self.dist_M[pos_nodes],dim=1)<torch.norm(self.dist_M[src_nodes]-self.dist_M[neg_nodes],dim=1)
        sim_label= torch.norm(self.X[src_nodes]-self.X[pos_nodes],dim=1)<torch.norm(self.X[src_nodes]-self.X[neg_nodes],dim=1)
        sim_loss = (node_emb[src_nodes]*node_emb[pos_nodes]).sum(dim=1) - (node_emb[src_nodes]*node_emb[neg_nodes]).sum(dim=1)
        sim_label = sim_label*2-1


        # y = torch.ones(node_emb[pos_nodes].shape[0]).to(self.X.device)
        # loss = nn.CosineEmbeddingLoss()
        # return loss(node_emb[src_nodes], node_emb[pos_nodes], y) + loss(node_emb[src_nodes], node_emb[neg_nodes], -1 * y)

        return F.relu(sim_loss * -sim_label + self.sim_loss_m).mean()

    def train_batch(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.sigma*self.ae_loss() + self.beta*self.tf_loss() + self.gamma*self.sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_sage(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.gamma*self.sage_sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_sage_decoder(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.sigma*self.ae_loss() + self.beta*self.tf_loss() + self.gamma*self.sage_decoder_sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def train_gcn(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.gamma*self.gcn_sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_gcn_decoder(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.sigma*self.ae_loss() + self.beta*self.tf_loss() + self.gamma*self.gcn_decoder_sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def train_gat(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.gamma*self.gat_sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_gat_decoder(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.sigma*self.ae_loss() + self.beta*self.tf_loss() + self.gamma*self.gat_decoder_sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_cheb(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.gamma*self.cheb_sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_cheb_decoder(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.sigma*self.ae_loss() + self.beta*self.tf_loss() + self.gamma*self.cheb_decoder_sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_node2vec(self):
        self.node2vec.train()
        self.optimizer.zero_grad()
        loss = self.gamma*self.node2vec_sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_node2vec_decoder(self):
        self.node2vec.train()
        self.optimizer.zero_grad()
        loss = self.sigma*self.ae_loss() + self.beta*self.tf_loss() + self.gamma*self.node2vec_decoder_sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_bert(self):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.gamma*self.bert_sim_loss()
        # loss = self.sigma*self.bert_ae_loss() + self.gamma*self.bert_sim_loss()
        # loss = self.bert_ae_loss() + self.bert_sim_loss()
        # loss = self.sigma*self.ae_loss() + self.beta*self.tf_loss() + self.gamma*self.sim_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, qX, ans, topk, verbose=False):
        self.model.eval()

        node_emb = self.model.forward(self.X, self.edge_index)
        
        q_emb = self.model.encoder(qX)

        avg_hit = eval_Z(node_emb, q_emb, ans, k=topk, verbose=verbose)

        return avg_hit

    def sage_test(self, qX, ans, topk, verbose=False):
        self.model.eval()

        node_emb = self.model.sage_forward(self.X, self.edge_index)
        
        q_emb = self.model.encoder(qX)

        avg_hit = eval_Z(node_emb, q_emb, ans, k=topk, verbose=verbose)

        return avg_hit

    def gcn_test(self, qX, ans, topk, verbose=False):
        self.model.eval()

        node_emb = self.model.gcn_forward(self.X, self.edge_index)
        
        q_emb = self.model.encoder(qX)

        avg_hit = eval_Z(node_emb, q_emb, ans, k=topk, verbose=verbose)

        return avg_hit

    def gat_test(self, qX, ans, topk, verbose=False):
        self.model.eval()

        node_emb = self.model.gat_forward(self.X, self.edge_index)
        
        q_emb = self.model.encoder(qX)

        avg_hit = eval_Z(node_emb, q_emb, ans, k=topk, verbose=verbose)

        return avg_hit


    def cheb_test(self, qX, ans, topk, verbose=False):
        self.model.eval()

        node_emb = self.model.cheb_forward(self.X, self.edge_index)
        
        q_emb = self.model.encoder(qX)

        avg_hit = eval_Z(node_emb, q_emb, ans, k=topk, verbose=verbose)

        return avg_hit


    def node2vec_test(self, qX, ans, topk, verbose=False):
        self.node2vec.eval()

        node_emb = self.node2vec.forward()
        
        q_emb = self.model.encoder(qX)

        avg_hit = eval_Z(node_emb, q_emb, ans, k=topk, verbose=verbose)

        return avg_hit

    def bert_test(self, qX, ans, topk, verbose=False):
        self.model.eval()
        device = self.X.device
       
        node_emb = self.model.bert_forward(self.X, self.edge_index, self.raw_embeddings, self.wl_embedding, self.int_embeddings, self.hop_embeddings)
        
        null_emb = torch.torch.zeros_like(qX).to(device)
        qX_reshaped = torch.reshape(qX, (qX.shape[0],1, qX.shape[1]))
        q_raw_emb =  torch.cat((qX_reshaped, qX_reshaped, qX_reshaped), dim=1).type(torch.float32).to(device)

        q_emb = self.model.bert_forward(qX, torch.torch.zeros_like(self.edge_index, dtype=torch.int64).to(device), q_raw_emb, torch.torch.zeros(len(qX), 3, dtype=torch.int64).to(device), torch.torch.zeros(len(qX), 3, dtype=torch.int64).to(device), torch.torch.zeros(len(qX), 3, dtype=torch.int64).to(device))

        avg_hit = eval_Z(node_emb, q_emb, ans, k=topk, verbose=verbose)

        return avg_hit

    def save(self, dir):
        if dir is not None:
            torch.save(self.model.state_dict(), dir)

    def decay_learning_rate(self, epoch, init_lr):
        lr = init_lr / (1 + 0.05 * epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer


