:math:`\mu\text{TC}` Command Line Interface
===========================================

For any given text classification task, :math:`\mu\text{TC}` will try to find a suitable text model from a set of possible models defined in the configuration space using the command:

.. code:: bash
	  
	  microTC-params -k3  -Smacrorecall -s24 -n24 user-profiling.json -o user-profiling.params

the parameters means for:

- `user-profiling.json` is database of exemplars, one json-dictionary per line with text and klass keywords
- `-k3` three folds
- `-s24` specifies that the parameter space should be sampled in 24 points and then get the best among them
- `-n24` let us specify the number of processes to be launch, it is a good idea to set `-s` as a multiply of `-n`.
- `-o user-profiling.params` specifies the file to store the configurations found by the parameter selection process, in best first order
- `-S` or `--score` the name of the fitness function (e.g., macrof1, microf1, macrorecall, accuracy, r2, pearsonr, spearmanr)
- `-H` makes b4msa to perform last hill climbing search for the parameter selection, in many cases, this will produce much better configurations (never worst, guaranteed)
- all of these parameters have default values, such as no arguments are needed

Notes:
- "text"  can be a string or an array of strings, in the last case, the final vector considers each string as independent strings.
- there is no typo, we use "klass" instead of "class" because of oscure historical reasons
- `-k` accepts an a:b syntax that allow searching in a sample of size a and test in b; for 0 < a < 1, and 0 < b < 1. It is common to use b = 1 - a; however, it is not a hard constraint and just you need to follow `a + b <= 1` and no overlapping ranges.
- If `-S` is `r2`, `pearsonr`, or `spearmanr` then :math:`\mu\text{TC}` computes the parameters for a regression task.


Training the model
======================

At this point, we are in the position to train a model.
Let us that the workload is `emotions.json` and that the parameters are in
`emotions.params` then the following command will save the model in `emotions.model`

.. code:: bash

	  microtc-train -o emptions.model -m emotions.params emotions.json

You can create a regressor adding the `-R` option to `microtc-train`

Using the model
=======================

At this point, we are in the position to test the model (i.e,
`emotions.model`) in a new set. That is, we are in the position to ask
the classifier to assign a label to a particular text.

.. code:: bash
   
   microtc-predict -m emotions.model -o emotions-predicted.json test-emotions.json

Finally, you can evaluate the performance of the prediction as follows:

.. code:: bash

	  microtc-perf gold.json emotions-predicted.json

This will show a number of scores in the screen.

.. code:: json

    {
    "accuracy": 0.7025,
    "f1_anger": 0.705,
    "f1_fear": 0.6338797814207651,
    "f1_joy": 0.7920353982300885,
    "f1_sadness": 0.6596858638743456,
    "macrof1": 0.6976502608812997,
    "macrof1accuracy": 0.490099308269113,
    "macrorecall": 0.7024999999999999,
    "microf1": 0.7025,
    "quadratic_weighted_kappa": 0.5773930753564155
    }


or, in case of provide the `--regression` flag

.. code:: json

    {
    "filename": "some-path/some-name.predicted",
    "pearsonr": [
        0.6311471948385253,
        1.2734619266038659e-23
    ],
    "r2": 0.3276512897198096,
    "spearmanr": [
        0.6377984613587965,
        3.112636137077516e-24
    ]
    }

