"""Quant demo v0.1 contract implementation.

The package deliberately separates read-only snapshot projection from the
explicit mutation services.  Viewer modules never import builders or network
clients.
"""
