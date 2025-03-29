import scipy as sp
from tqdm import tqdm
import numpy as np
import torch
import os
import random
import scipy.sparse as sp

from elliot.utils.write import store_recommendation
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .SVDGCNModel import SVDGCNModel


class SVDGCN(RecMixin, BaseRecommenderModel):
    r"""
    SVD-GCN: A Simplified Graph Convolution Paradigm for Recommendation

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        coef_u: User-user coefficient
        coef_i: Item-item coefficient
        beta: Beta parameter for weighting function
        req_vec: No description
        alpha: For the R normalization
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_coef_u", "coef_u", "coef_u", 0.1, float, None),
            ("_coef_i", "coef_i", "coef_i", 0.1, float, None),
            ("_alpha", "alpha", "alpha", 1, float, None),
            ("_beta", "beta", "beta", 0.1, float, None),
            ("_req_vec", "req_vec", "req_vec", 90, int, None)
        ]
        self.autoset_params()

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # fake_test_dict = {u: {i: 0 for i in data.test_dict[u]} for u in data.test_dict}
        # full_dataset = {user_train: {**fake_test_dict[user_train], **data.train_dict[user_train]}  # Unisci i dizionari interni con **
        # for user_train in data.train_dict
        #     if user_train in fake_test_dict}
        #
        # items_total = ({k for a in data.train_dict.values() for k in a.keys()}
        #                .union({k for a in fake_test_dict.values() for k in a.keys()}))
        # num_items_total = len(items_total)
        # data.private_items = {p: i for p, i in enumerate(items_total)}
        # data.public_items = {v: k for k, v in data.private_items.items()}
        #
        # i_full_dataset_dict = {data.public_users[user]: {data.public_items[i]: v for i, v in items.items() if v!=0 }
        #                      for user, items in full_dataset.items()}
        # rows_cols = [(u, i) for u, items in i_full_dataset_dict.items() for i in items.keys()]
        # rows = [u for u, _ in rows_cols]
        # cols = [i for _, i in rows_cols]
        # i_full_dataset_dict_sparse = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
        #               shape=(len(data.users), len(items_total)))
        # self.rate_matrix = i_full_dataset_dict_sparse.toarray()
        # self.mask = np.where((i_full_dataset_dict_sparse.toarray() == 0), True, False)


        self.rate_matrix = torch.from_numpy(data.sp_i_train.todense())
        # self.rate_matrix = torch.from_numpy(self.rate_matrix)
        self.rate_matrix.to(self.device)

        if self._config.data_config.strategy == 'fixed':
            path = os.path.split(self._config.data_config.train_path)[0]
            if not (os.path.exists(path + f'/svd_u_{self._alpha}.npy')
                    or os.path.exists(path + f'/svd_i_{self._alpha}.npy') or os.path.exists(
                    path + f'/svd_value_{self._alpha}.npy')):
                self.logger.info(
                    f"Processing singular values as they haven't been calculated before on this dataset...")
                U, value, V = self.preprocess(path, data.num_users, data.num_items)
                self.logger.info(f"Processing end!")
            else:
                self.logger.info(f"Singular values have already been processed for this dataset!")
                value = torch.Tensor(np.load(path + f'/svd_value_{self._alpha}.npy'))
                U = torch.Tensor(np.load(path + f'/svd_u_{self._alpha}.npy'))
                V = torch.Tensor(np.load(path + f'/svd_v_{self._alpha}.npy'))
        else:
            raise NotImplementedError('The check when strategy is different from fixed has not been implemented yet!')

        self.user_matrix = ((self.rate_matrix.mm(self.rate_matrix.t())) != 0).float().to(self.device)
        self.item_matrix = ((self.rate_matrix.t().mm(self.rate_matrix)) != 0).float().to(self.device)

        self._model = SVDGCNModel(
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            l_w=self._l_w,
            coef_u=self._coef_u,
            coef_i=self._coef_i,
            beta=self._beta,
            req_vec=self._req_vec,
            u=U,
            value=value,
            v=V,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "SVDGCN" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def preprocess(self, dataset, users, items):
        D_u = self.rate_matrix.sum(1) + self._alpha
        D_i = self.rate_matrix.sum(0) + self._alpha

        for i in range(users):
            if D_u[i] != 0:
                D_u[i] = 1 / D_u[i].sqrt()

        for i in range(items):
            if D_i[i] != 0:
                D_i[i] = 1 / D_i[i].sqrt()

        # \tilde{R}
        rate_matrix = D_u.unsqueeze(1) * self.rate_matrix * D_i

        # free space
        del D_u, D_i

        U, value, V = torch.svd_lowrank(rate_matrix, q=400, niter=30)

        np.save(dataset + f'/svd_u_{self._alpha}.npy', U.cpu().numpy())
        np.save(dataset + f'/svd_v_{self._alpha}.npy', V.cpu().numpy())
        np.save(dataset + f'/svd_value_{self._alpha}.npy', value.cpu().numpy())

        return U, value, V

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for _, _ in enumerate(range(0, int(self._data.transactions // self._batch_size))):
                    steps += 1
                    u = np.random.randint(0, self._num_users, self._batch_size)
                    p = torch.multinomial(self.rate_matrix[u], 1, True).squeeze(1)
                    nega = torch.multinomial(1 - self.rate_matrix[u], 1, True).squeeze(1)
                    up = torch.multinomial(self.user_matrix[u], 1, True).squeeze(1)
                    un = torch.multinomial(1 - self.user_matrix[u], 1, True).squeeze(1)
                    pp = torch.multinomial(self.item_matrix[p], 1, True).squeeze(1)
                    pn = torch.multinomial(1 - self.item_matrix[p], 1, True).squeeze(1)
                    batch = u, p, nega, up, un, pp, pn
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / steps)

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self.get_users_rating(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_users_rating(self, batch_start, batch_stop):
        final_user = self._model.user_vector[batch_start: batch_stop].mm(self._model.FS)
        final_item = self._model.item_vector.mm(self._model.FS)
        return (final_user.mm(final_item.t())).sigmoid().to(self.device) - (self.rate_matrix[batch_start: batch_stop] * 1000).to(self.device)

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss :.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False
