"""One numeric preparation recipe shared by coordinator and freeze producer.

Saved native deltas remain readable independently of this current recipe.
Only new preparation/cache admission compares the current recipe identity.
"""

RECIPE = 'native-reference-appearance-freeze-v7-preserve-source-hair'
