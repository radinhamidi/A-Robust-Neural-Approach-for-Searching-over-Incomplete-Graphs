import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from DatasetLoader import DatasetLoader
from Trainer import *
from utils import *
from models import *
from args import *
from MethodBertComp import GraphBertConfig
import random

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)



if __name__ == "__main__":

    args = make_args()

    print(args)

    # verbose = True #
    verbose = False #
    

    device = torch.device(f"cuda:{args.cuda}")

    model_path = get_saved_model_path(args)

    print("Model will be saved at ", model_path)

    X, edge_index = load_data(args)
    X = coo2torch(X).to_dense()
    edge_index = torch.tensor(edge_index)
    nodes_num, attr_num = X.shape

    print(
        f"{args.dataset}: nodes_num:{nodes_num}, attr_num:{attr_num}, edges_num:{edge_index.shape[1]}"
    )

    topk = args.topk

    kw = f"kw{args.kw}"

    gt = load_gt(f"./data/{args.dataset}/test_gt/{kw}/{kw}_top{str(topk)}_output.txt")

    val_gt = load_gt(f"./data/{args.dataset}/val_gt/{args.hw}%hidden_graph/{kw}/{kw}_top{str(topk)}_output.txt")

    qs = []
    ans = []
    for q in gt.keys():
        qs.append(str2int(q))
        ans.append(gt[q])

    val_qs = []
    val_ans = []
    for q in val_gt.keys():
        val_qs.append(str2int(q))
        val_ans.append(val_gt[q])

    # -------------#

    qX = queries2tensor(qs, attr_num).to(device)
    val_qX = queries2tensor(val_qs, attr_num).to(device)

    X = X.to(device)
    edge_index = edge_index.to(device)

    output_dict = {}

###################
    for repeat in range(args.repeat):
        args.d = 128
        args.hid_d = 512
        k = 2
        nfeature = 3703
        ngraph = 3327
        nclass = 6
        num_attention_heads = 4
        num_hidden_layers = 2
        residual_type = 'graph_raw'
        bert_config = GraphBertConfig(residual_type = residual_type, k=k, x_size=nfeature, y_size=nclass, hidden_size=args.d, intermediate_size=args.hid_d, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)

        data_obj = DatasetLoader()
        dataset_name = "citeseer"
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
        data_obj.dataset_name = dataset_name
        data_obj.k = k
        data_obj.device = device
        data_obj.load_all_tag = True
        data_o = data_obj.load()
        raw_embeddings = data_o['raw_embeddings']
        wl_embedding = data_o['wl_embedding']
        int_embeddings = data_o['int_embeddings']
        hop_embeddings = data_o['hop_embeddings']
        
        raw_embeddings = torch.tensor(raw_embeddings).to(device)
        wl_embedding = torch.tensor(wl_embedding).to(device)
        int_embeddings = torch.tensor(int_embeddings).to(device)
        hop_embeddings = torch.tensor(hop_embeddings).to(device)

        model = KSNN(
            X.shape[1],
            args.hid_d,
            args.d,
            config=bert_config,
            layer_num=args.layer_num,
            conv_num=args.conv_num,
            alpha=args.alpha,
        ).to(device)

        trainer = Trainer(model, X, edge_index, args, raw_embeddings, wl_embedding, int_embeddings, hop_embeddings)

        for e in range(args.epochs_num):

            trainer.train_bert()

            if e % 5 == 0:
                trainer.decay_learning_rate(e, args.lr)

            if e % args.eval_time == 0:
                hit_valid = trainer.bert_test(val_qX, val_ans, topk)
                if hit_valid <= trainer.bestResult:
                    time += 1
                else:
                    trainer.bestResult = hit_valid
                    time = 0
                    trainer.save(model_path)
                if time > 20:
                    print(("BEST RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                    break

        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))

        trainer.model = model
        output_dict["Proposed Method"] = output_dict.get('Proposed Method',[]) + [trainer.bert_test(qX, ans, topk, verbose=verbose)]
        print(f'repeat {repeat}: {output_dict["Proposed Method"][-1]}')


    output_dict["Proposed Method"] = np.array(output_dict["Proposed Method"]).mean()
    print(f"Test Result:{output_dict['Proposed Method']:.4}")    


    args = make_args()
    for repeat in range(args.repeat):
    
        model = KSNN(
            X.shape[1],
            args.hid_d,
            args.d,
            layer_num=args.layer_num,
            conv_num=args.conv_num,
            alpha=args.alpha,
        ).to(device)

        trainer = Trainer(model, X, edge_index, args)

        for e in range(args.epochs_num):
            trainer.train_batch()

            if e % 5 == 0:
                trainer.decay_learning_rate(e, args.lr)

            if e % args.eval_time == 0:

                hit_valid = trainer.test(val_qX, val_ans, topk)

                if hit_valid <= trainer.bestResult:
                    time += 1
                else:
                    trainer.bestResult = hit_valid
                    time = 0
                    trainer.save(model_path)
                if time > 20:
                    print(("BEST RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                    break

        

        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))

        trainer.model = model

        output_dict["KS-GNN"] = output_dict.get('KS-GNN',[]) + [trainer.test(qX, ans, topk, verbose=verbose)]

        print(f'repeat {repeat}: {output_dict["KS-GNN"][-1]}')

    
    output_dict["KS-GNN"] = np.array(output_dict["KS-GNN"]).mean()
    k_model = model

    print(f"Test Result:{output_dict['KS-GNN']:.4}")
############################

    if args.pca_verbal:

        for repeat in range(args.repeat):

            model = KSNN(
                X.shape[1],
                args.hid_d,
                args.d,
                layer_num=args.layer_num,
                conv_num=args.conv_num,
                alpha=args.alpha,
            ).to(device)

            trainer = Trainer(model, X, edge_index, args)

            for e in range(args.epochs_num):

                trainer.train_gcn()

                if e % 10 == 0:
                    trainer.decay_learning_rate(e, args.lr)

                if e % args.eval_time == 0:
                    hit_valid = trainer.gcn_test(val_qX, val_ans, topk)
                    if hit_valid <= trainer.bestResult:
                        time += 1
                    else:
                        trainer.bestResult = hit_valid
                        time = 0
                        trainer.save(model_path)
                    if time > 20:
                        print(("BEST RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                        break

            if os.path.isfile(model_path):
                model.load_state_dict(torch.load(model_path))

            trainer.model = model
            output_dict["GCN"] = output_dict.get('GCN',[]) + [trainer.gcn_test(qX, ans, topk, verbose=verbose)]
            print(f'repeat {repeat}: {output_dict["GCN"][-1]}')


        output_dict["GCN"] = np.array(output_dict["GCN"]).mean()
        print(f"Test Result:{output_dict['GCN']:.4}")
        

        for repeat in range(args.repeat):

            model = KSNN(
                X.shape[1],
                args.hid_d,
                args.d,
                layer_num=args.layer_num,
                conv_num=args.conv_num,
                alpha=args.alpha,
            ).to(device)

            trainer = Trainer(model, X, edge_index, args)

            for e in range(args.epochs_num):

                trainer.train_gcn_decoder()

                if e % 10 == 0:
                    trainer.decay_learning_rate(e, args.lr)

                if e % args.eval_time == 0:
                    hit_valid = trainer.gcn_test(val_qX, val_ans, topk)
                    if hit_valid <= trainer.bestResult:
                        time += 1
                    else:
                        trainer.bestResult = hit_valid
                        time = 0
                        trainer.save(model_path)
                    if time > 20:
                        print(("BEST RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                        break

            if os.path.isfile(model_path):
                model.load_state_dict(torch.load(model_path))

            trainer.model = model
            output_dict["GCN+Decoder"] = output_dict.get('GCN+Decoder',[]) + [trainer.gcn_test(qX, ans, topk, verbose=verbose)]
            print(f'repeat {repeat}: {output_dict["GCN+Decoder"][-1]}')


        output_dict["GCN+Decoder"] = np.array(output_dict["GCN+Decoder"]).mean()
        print(f"Test Result:{output_dict['GCN+Decoder']:.4}")


        for repeat in range(args.repeat):

            model = KSNN(
                X.shape[1],
                args.hid_d,
                args.d,
                layer_num=args.layer_num,
                conv_num=args.conv_num,
                alpha=args.alpha,
            ).to(device)

            trainer = Trainer(model, X, edge_index, args)

            for e in range(args.epochs_num):

                trainer.train_gat()

                if e % 10 == 0:
                    trainer.decay_learning_rate(e, args.lr)

                if e % args.eval_time == 0:
                    hit_valid = trainer.gat_test(val_qX, val_ans, topk)
                    if hit_valid <= trainer.bestResult:
                        time += 1
                    else:
                        trainer.bestResult = hit_valid
                        time = 0
                        trainer.save(model_path)
                    if time > 20:
                        print(("BEST RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                        break

            if os.path.isfile(model_path):
                model.load_state_dict(torch.load(model_path))

            trainer.model = model
            output_dict["GAT"] = output_dict.get('GAT',[]) + [trainer.gat_test(qX, ans, topk, verbose=verbose)]
            print(f'repeat {repeat}: {output_dict["GAT"][-1]}')


        output_dict["GAT"] = np.array(output_dict["GAT"]).mean()
        print(f"Test Result:{output_dict['GAT']:.4}")


        for repeat in range(args.repeat):

            model = KSNN(
                X.shape[1],
                args.hid_d,
                args.d,
                layer_num=args.layer_num,
                conv_num=args.conv_num,
                alpha=args.alpha,
            ).to(device)

            trainer = Trainer(model, X, edge_index, args)

            for e in range(args.epochs_num):

                trainer.train_gat_decoder()

                if e % 10 == 0:
                    trainer.decay_learning_rate(e, args.lr)

                if e % args.eval_time == 0:
                    hit_valid = trainer.gat_test(val_qX, val_ans, topk)
                    if hit_valid <= trainer.bestResult:
                        time += 1
                    else:
                        trainer.bestResult = hit_valid
                        time = 0
                        trainer.save(model_path)
                    if time > 20:
                        print(("BEST RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                        break

            if os.path.isfile(model_path):
                model.load_state_dict(torch.load(model_path))

            trainer.model = model
            output_dict["GAT+Decoder"] = output_dict.get('GAT+Decoder',[]) + [trainer.gat_test(qX, ans, topk, verbose=verbose)]
            print(f'repeat {repeat}: {output_dict["GAT+Decoder"][-1]}')


        output_dict["GAT+Decoder"] = np.array(output_dict["GAT+Decoder"]).mean()
        print(f"Test Result:{output_dict['GAT+Decoder']:.4}")


        for repeat in range(args.repeat):

            model = KSNN(
                X.shape[1],
                args.hid_d,
                args.d,
                layer_num=args.layer_num,
                conv_num=args.conv_num,
                alpha=args.alpha,
            ).to(device)

            trainer = Trainer(model, X, edge_index, args)

            for e in range(args.epochs_num):

                trainer.train_cheb()

                if e % 10 == 0:
                    trainer.decay_learning_rate(e, args.lr)

                if e % args.eval_time == 0:
                    hit_valid = trainer.cheb_test(val_qX, val_ans, topk)
                    if hit_valid <= trainer.bestResult:
                        time += 1
                    else:
                        trainer.bestResult = hit_valid
                        time = 0
                        trainer.save(model_path)
                    if time > 20:
                        print(("BEST RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                        break

            if os.path.isfile(model_path):
                model.load_state_dict(torch.load(model_path))

            trainer.model = model
            output_dict["ChebConv"] = output_dict.get('ChebConv',[]) + [trainer.cheb_test(qX, ans, topk, verbose=verbose)]
            print(f'repeat {repeat}: {output_dict["ChebConv"][-1]}')


        output_dict["ChebConv"] = np.array(output_dict["ChebConv"]).mean()
        print(f"Test Result:{output_dict['ChebConv']:.4}")


        for repeat in range(args.repeat):

            model = KSNN(
                X.shape[1],
                args.hid_d,
                args.d,
                layer_num=args.layer_num,
                conv_num=args.conv_num,
                alpha=args.alpha,
            ).to(device)

            trainer = Trainer(model, X, edge_index, args)

            for e in range(args.epochs_num):

                trainer.train_cheb_decoder()

                if e % 10 == 0:
                    trainer.decay_learning_rate(e, args.lr)

                if e % args.eval_time == 0:
                    hit_valid = trainer.cheb_test(val_qX, val_ans, topk)
                    if hit_valid <= trainer.bestResult:
                        time += 1
                    else:
                        trainer.bestResult = hit_valid
                        time = 0
                        trainer.save(model_path)
                    if time > 20:
                        print(("BEST RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                        break

            if os.path.isfile(model_path):
                model.load_state_dict(torch.load(model_path))

            trainer.model = model
            output_dict["ChebConv+Decoder"] = output_dict.get('ChebConv+Decoder',[]) + [trainer.cheb_test(qX, ans, topk, verbose=verbose)]
            print(f'repeat {repeat}: {output_dict["ChebConv+Decoder"][-1]}')


        output_dict["ChebConv+Decoder"] = np.array(output_dict["ChebConv+Decoder"]).mean()
        print(f"Test Result:{output_dict['ChebConv+Decoder']:.4}")


        for repeat in range(args.repeat):

            model = KSNN(
                X.shape[1],
                args.hid_d,
                args.d,
                layer_num=args.layer_num,
                conv_num=args.conv_num,
                alpha=args.alpha,
            ).to(device)

            trainer = Trainer(model, X, edge_index, args)

            for e in range(args.epochs_num):

                trainer.train_sage()

                if e % 10 == 0:
                    trainer.decay_learning_rate(e, args.lr)

                if e % args.eval_time == 0:
                    hit_valid = trainer.sage_test(val_qX, val_ans, topk)
                    if hit_valid <= trainer.bestResult:
                        time += 1
                    else:
                        trainer.bestResult = hit_valid
                        time = 0
                        trainer.save(model_path)
                    if time > 20:
                        print(("BEST RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                        break

            if os.path.isfile(model_path):
                model.load_state_dict(torch.load(model_path))

            trainer.model = model
            output_dict["GraphSAGE"] = output_dict.get('GraphSAGE',[]) + [trainer.sage_test(qX, ans, topk, verbose=verbose)]
            print(f'repeat {repeat}: {output_dict["GraphSAGE"][-1]}')


        output_dict["GraphSAGE"] = np.array(output_dict["GraphSAGE"]).mean()
        print(f"Test Result:{output_dict['GraphSAGE']:.4}")


        if args.he == 0:
            
            for repeat in range(args.repeat):
        
                model = KSNN(
                    X.shape[1],
                    args.hid_d,
                    args.d,
                    layer_num=args.layer_num,
                    conv_num=args.conv_num,
                    alpha=args.alpha,
                ).to(device)

                trainer = Trainer(model, X, edge_index, args)

                for e in range(args.epochs_num):
                    trainer.train_node2vec()

                    if e % 5 == 0:
                        trainer.decay_learning_rate(e, args.lr)

                    if e % args.eval_time == 0:

                        hit_valid = trainer.node2vec_test(val_qX, val_ans, topk)

                        if hit_valid <= trainer.bestResult:
                            time += 1
                        else:
                            trainer.bestResult = hit_valid
                            time = 0
                            trainer.save(model_path)
                        if time > 20:
                            print(("BEST RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                            break
                
                if os.path.isfile(model_path):
                    model.load_state_dict(torch.load(model_path))

                trainer.model = model

                output_dict["Node2Vec"] = output_dict.get('Node2Vec',[]) + [trainer.node2vec_test(qX, ans, topk, verbose=verbose)]

                print(f'repeat {repeat}: {output_dict["Node2Vec"][-1]}')

            
            output_dict["Node2Vec"] = np.array(output_dict["Node2Vec"]).mean()
            k_model = model

            print(f"Test Result:{output_dict['Node2Vec']:.4}")


            for repeat in range(args.repeat):
            
                model = KSNN(
                    X.shape[1],
                    args.hid_d,
                    args.d,
                    layer_num=args.layer_num,
                    conv_num=args.conv_num,
                    alpha=args.alpha,
                ).to(device)

                trainer = Trainer(model, X, edge_index, args)

                for e in range(args.epochs_num):
                    trainer.train_node2vec_decoder()

                    if e % 5 == 0:
                        trainer.decay_learning_rate(e, args.lr)

                    if e % args.eval_time == 0:

                        hit_valid = trainer.node2vec_test(val_qX, val_ans, topk)

                        if hit_valid <= trainer.bestResult:
                            time += 1
                        else:
                            trainer.bestResult = hit_valid
                            time = 0
                            trainer.save(model_path)
                        if time > 20:
                            print(("BEST RESULT ON VALIDATE DATA:{:.4}").format(trainer.bestResult))
                            break

                if os.path.isfile(model_path):
                    model.load_state_dict(torch.load(model_path))

                trainer.model = model

                output_dict["Node2Vec + Decoder"] = output_dict.get('Node2Vec + Decoder',[]) + [trainer.node2vec_test(qX, ans, topk, verbose=verbose)]

                print(f'repeat {repeat}: {output_dict["Node2Vec + Decoder"][-1]}')

            
            output_dict["Node2Vec + Decoder"] = np.array(output_dict["Node2Vec + Decoder"]).mean()
            k_model = model

            print(f"Test Result:{output_dict['Node2Vec + Decoder']:.4}")


        output_dict["Conv-PCA"] = eval_Z(
            trainer.model.pca(X, edge_index),
            qX @ trainer.model.pca_v,
            ans,
            k=topk,
            verbose=verbose,
        )
        print(f"Conv-PCA:{output_dict['Conv-PCA']:.4}")

        output_dict["KS-PCA"] = eval_Z(
            trainer.model.kspca(X, edge_index),
            qX @ trainer.model.pca_v,
            ans,
            k=topk,
            verbose=verbose,
        )
        print(f"KS-PCA:{output_dict['KS-PCA']:.4}")

        output_dict["PCA"] = eval_PCA(X, qX, ans, k=topk, verbose=verbose)

        print(f"PCA:{output_dict['PCA']:.4}")

    for key in output_dict:
        output_dict[key] = round(output_dict[key]*100,2)

    print(output_dict)
