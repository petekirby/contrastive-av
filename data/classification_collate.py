# Responsibility: Eric and/or Henry (you will both need something like this)

# This file needs to be filled out to get pairs for classification from the training data.
# ClassificationCollator should be implemented, kind of like ContrastiveCollator in contrastive_collate.py
# ClassificationPairCollator can also be implemented but should be very similar to the pair collator in contrastive_collate.py
# because the pairs are already constructed for the pair collator.
# ClassificationCollator will need to produce pairs.
# You should assume the batch is already processed somehow, like with MPerClass, to have at least 1 positive (same author) for each 'anchor' (text/doc).

# Recommendation:
#   from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
#   a, p, n = lmu.get_random_triplet_indices(labels, t_per_anchor=1)
# This will do all the logic of getting 1 positive and 1 negative (at random) for each 'anchor'.
# Then you have to split this into (a, p) positive examples and (a, n) negative examples (and do any other processing you want).
# For more on this function, see the test code and source code:
# https://github.com/KevinMusgrave/pytorch-metric-learning/blob/c8350998ebc8aacf2c45de50e2556bc854cc0361/tests/utils/test_loss_and_miner_utils.py#L167 
# https://github.com/KevinMusgrave/pytorch-metric-learning/blob/c8350998ebc8aacf2c45de50e2556bc854cc0361/src/pytorch_metric_learning/utils/loss_and_miner_utils.py#L140

# Integration:
# You will want to use these functions in pan_data.py based on
#         if isinstance(model, YourLightningModule):
# to set self.collate_fn and self.pair_collate_fn
# You will also need to make sure you use MPerClass or similar sampler so that each anchor has at least 1 positive.
