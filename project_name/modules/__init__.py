from typing import Literal

from pydantic import BaseModel, Field

from .net1 import Net1, Net1Cfg


class Net2Cfg(BaseModel):
    name: Literal["net2"]


class NetCfg(BaseModel):
    specific: Net1Cfg | Net2Cfg = Field(..., discriminator="name")


def get_module(cfg: NetCfg):
    if cfg.specific.name == "net1":
        net = Net1(cfg.specific)
    else:
        raise ValueError(f"Unexpected Module: name={cfg.specific.name}")
    return net
