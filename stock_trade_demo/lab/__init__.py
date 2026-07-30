"""Isolated Learn/Lab-1 domain.

This package deliberately does not import ``web.state``, strategy registries,
or any data loader.  The Web layer injects one verified materialized snapshot
into the worker contract.
"""

from lab.service import LabService

__all__ = ['LabService']
