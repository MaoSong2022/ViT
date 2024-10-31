from torch import nn
from typing import Optional, Dict
import torch
import torch.nn.functional as F


class ViTConfig:

    model_type = "vit"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        layer_norm_eps: float = 1e-12,
        hidden_activation: str = "gelu",
        hidden_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        qkv_bias: bool = True,
        encoder_stride: int = 16,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_activation = hidden_activation
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride


class ViTPatchEmbedding(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()

        num_channels = config.num_channels
        hidden_size = config.hidden_size
        if isinstance(config.image_size, int):
            image_size = (config.image_size, config.image_size)
        else:
            image_size = config.image_size

        if isinstance(config.patch_size, int):
            patch_size = (config.patch_size, config.patch_size)
        else:
            patch_size = config.patch_size

        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        # check if num_channels are equal to self.num_channels
        # if interpolate_pos_encoding is False,
        # check if height==self.image_size[0] and width==self.image_size[1]
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ViTEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig, use_mask_token: bool = False):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            if use_mask_token
            else None
        )
        self.patch_embedding = ViTPatchEmbedding(config)
        num_patches = self.patch_embedding.num_patches

        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_rate)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_embedding(
        self, embeddings: torch.Tensor, width: int, height: int
    ):
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        if (
            not torch.jit.is_tracing()
            and num_positions == num_patches
            and width == height
        ):
            return self.position_embeddings

        cls_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = (num_positions**0.5).to(torch.int64)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat((cls_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        """Generate embeddings for a given image that is represented as pixel_values.

        Args:
            pixel_values (torch.Tensor): a tensor representing an image whose
                shape: (batch_size, num_channels, height, width)
            bool_masked_pos (Optional[torch.Tensor], optional): whether to use mask on pixel values.
                Defaults to None.
            interpolate_pos_encoding (bool, optional): whether to use interpolate pos embedding.
                Defaults to False.

        Returns:
            torch.Tensor: the embedding of the image
        """
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embedding(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        if bool_masked_pos is not None:
            seq_len = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add [cls] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional embeddings to the embedded patch tokens
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_embedding(
                embeddings, width, height
            )
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        assert (
            config.hidden_size % config.num_heads == 0
        ), "hidden_size must be a multiple of the number of heads"
        self.head_size = int(config.hidden_size / config.num_heads)
        self.scale = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))

        self.query = nn.Linear(config.hidden_size, self.head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.head_size, bias=config.qkv_bias)

        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout_rate)

    def forward(
        self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # transpose query and key to [batch_size, num_heads, seq_len, head_size]
        query = (
            query.transpose(1, 2)
            .contiguous()
            .view(-1, query.size(1), self.num_heads, self.head_size)
        )
        key = (
            key.transpose(1, 2)
            .contiguous()
            .view(-1, key.size(1), self.num_heads, self.head_size)
        )

        # calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(2, 3)) / self.scale

        if head_mask is not None:
            attention_scores = attention_scores * head_mask

        attentions = F.softmax(attention_scores, dim=-1)

        attentions = self.attention_dropout(attentions)

        hidden_states = torch.matmul(attentions, value)
        hidden_states = hidden_states.contiguous().view(
            -1, hidden_states.size(1), hidden_states.size(2) * self.head_size
        )
        hidden_states = self.attention_output(hidden_states)

        return {
            "hidden_states": hidden_states,
            "attentions": attentions,
        }


class ViTOutputLayer(nn.Module):
    def __init__(self, config: ViTConfig):
        """
        Initialize ViTOutputLayer with given configuration.

        Parameters:
        ----------
        config : ViTConfig
            An instance of ViTConfig class containing configuration parameters.

        Attributes:
        ----------
        dense_layer1 : nn.Linear
            A linear layer that projects the input from hidden_size to intermediate_size.
        intermediate_activation : nn.Module
            An activation function applied after dense_layer1.
        dense_layer2 : nn.Linear
            A linear layer that projects the output of intermediate_activation from intermediate_size to hidden_size.
        dropout : nn.Dropout
            A dropout layer applied to the output of dense_layer2.
        """
        super().__init__()
        self.dense_layer1 = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_activation, str):
            self.intermediate_activation = getattr(nn, config.hidden_activation)()
        else:
            self.intermediate_activation = config.hidden_activation
        self.dense_layer2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_layer1(hidden_states)
        hidden_states = self.intermediate_activation(hidden_states)
        hidden_states = self.dense_layer2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTLayer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.layer_norm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = ViTAttention(config)
        self.layer_norm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.output_layer = ViTOutputLayer(config)

    def forward(
        self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        hidden_states = self.layer_norm_before(hidden_states)
        self_attention_outputs = self.attention(hidden_states, head_mask)
        attentions = self_attention_outputs["attentions"]

        # residual connection
        hidden_states = hidden_states + self_attention_outputs["hidden_state"]

        mlp_output = self.layer_norm_after(hidden_states)
        hidden_states = self.output_layer(mlp_output)

        # residual connection
        hidden_states = hidden_states + mlp_output

        return {
            "hidden_states": hidden_states,
            "attentions": attentions,
        }


class ViTEncoder(nn.Module):

    def __init__(self, config: ViTConfig):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList(
            [ViTLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        all_hidden_states = ()
        all_self_attentions = ()

        for i, layer in enumerate(self.layers):
            all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs["hidden_states"]

            all_self_attentions = all_self_attentions + layer_outputs["attentions"]

        # Add the last hidden state to the outputs.
        all_hidden_states = (hidden_states,) + all_hidden_states

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
        }


class ViTPooler(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.Tanh(config.hidden_activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        "we pool the model by simply taking the hidden state corresponding to the first token"
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class ViTModel:

    def __init__(
        self,
        config: ViTConfig,
        use_mask_token: bool = False,
    ):
        self.config = config
        self.use_mask_token = use_mask_token

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        bool_masked_pos: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_hidden_state = encoder_outputs["last_hidden_state"]
        last_hidden_state = self.layer_norm(last_hidden_state)
        pooled_output = self.pooler(last_hidden_state)

        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooled_output,
            "hidden_states": encoder_outputs["hidden_states"],
            "attentions": encoder_outputs.attentions,
        }
