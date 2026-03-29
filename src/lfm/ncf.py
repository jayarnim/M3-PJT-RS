import torch
import torch.nn as nn
from components.feature_store import FeatureStore
from components.base import BayesModel
from components.base import BayesModelOutput
from .layers.embedding import build as build_embedding_layer
from .layers.bam import BayesianAttentionModules
from .layers.pooling import build as build_pooling_layer
from .layers.combination import build as build_comb_layer
from .layers.matching import build as build_matching_layer
from .layers.prediction import ProjectionLayer


class NeuralCollaborativeFiltering(BayesModel):
    def __init__(
        self,
        feature_store: FeatureStore,
        num_users: int,
        num_items: int,
        num_genres: int,
        embedding_dim: int,
        hidden_dim: list,
        comb: str,
        pooling: str,
        score: str,
        sampler: str,
        param_q: float,
        param_p: float,
        beta: int,
        dropout: float=None,
    ):
        super().__init__(locals())

        self.feature_store = feature_store

        # EMBEDDINGS ==========
        self.embedding = build_embedding_layer(
            name="idx",
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
        )
        self.genre_emb = nn.Embedding(
            num_embeddings=num_genres+2, 
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # POOLING ==========
        components = dict(
            user=BayesianAttentionModules(
                score=score,
                sampler=sampler,
                dim=embedding_dim, 
                param_q=param_q,
                param_p=param_p,
                beta=beta,
                dropout=dropout,
            ),
            item=build_pooling_layer(
                name=pooling,
            ),
        )
        self.pooling = nn.ModuleDict(components)

        # COMBINATION ==========
        kwargs = dict(
            name=comb,
            dim=embedding_dim,
        )
        components = dict(
            user=build_comb_layer(**kwargs),
            item=build_comb_layer(**kwargs),
        )
        self.comb = nn.ModuleDict(components)

        # MATCHING ==========
        self.matching = build_matching_layer(
            name="ncf",
            embedding_dim=(
                embedding_dim*2
                if comb=="cat"
                else embedding_dim
            ),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=hidden_dim[-1],
        )

        # INIT. GENRE EMBEDDING
        self.init_embeddings()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # EMBEDDING ==========
        user_emb, item_emb = self.embedding(user_idx, item_idx)

        # POOLING FEATURES ==========
        features = self.feature_store(user_idx, item_idx)

        output = self.pooling["user"](
            q=user_emb,
            k=self.genre_emb.weight[2:],
            v=self.genre_emb.weight[2:],
        )

        item_pooled = self.pooling["item"](
            indices=features["genre"],
            embeddings=self.genre_emb.weight,
        )

        # COMBINATION ==========
        user_combined = self.comb["user"](user_emb, output.context)
        item_combined = self.comb["item"](item_emb, item_pooled)

        # MATCHING ==========
        X_pred = self.matching(user_combined, item_combined)

        return X_pred, output.kld

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> BayesModelOutput:
        X_pred, kld = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
        return BayesModelOutput(
            logit=logit, 
            kld=kld,
        )

    def init_embeddings(self):
        kwargs = dict(
            tensor=self.genre_emb.weight, 
            mean=0.0, 
            std=0.01,
        )
        nn.init.normal_(**kwargs)