"""
A wrapper around dspy.Predict that accepts an 'adapter' argument.

This allows per-module adapter configuration, which is not yet supported
by the standard dspy.Predict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy
from dspy.dsp.utils.settings import settings

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter
    from dspy.signatures.signature import Signature
    from dspy.utils.callback import BaseCallback


class Predict(dspy.Predict):
    """A dspy.Predict subclass that supports an explicit adapter instance.

    Usage:
        adapter = TemplateAdapter(...)
        pred = Predict(MySignature, adapter=adapter)
    """

    def __init__(
        self,
        signature: str | type[Signature],
        callbacks: list[BaseCallback] | None = None,
        adapter: Adapter | None = None,
        **config,
    ):
        super().__init__(signature, callbacks, **config)
        self.adapter = adapter

    def forward(self, **kwargs):
        # Local override: if we have an adapter, force it into settings context
        # just for this call.
        if self.adapter:
            with settings.context(adapter=self.adapter):
                return super().forward(**kwargs)
        
        return super().forward(**kwargs)

    async def aforward(self, **kwargs):
        if self.adapter:
            with settings.context(adapter=self.adapter):
                return await super().aforward(**kwargs)
        
        return await super().aforward(**kwargs)
