# MiniLM sensitivity execution note

The frozen protocol and implementation were committed at `49a221e` before the
MiniLM sensitivity analysis was run. The first execution completed feature
construction and model prediction but stopped before results were displayed or
serialized because `stage3_evaluate.evaluate_model` returns a `(result,
predictions)` tuple. The sensitivity script had assigned that tuple directly to
the JSON results object, which is not serializable.

The subsequent code change only unpacks the existing return value and reuses its
prediction vector. No input, feature, representation, graph, split, classifier,
threshold, bootstrap, comparison, or reporting rule changed. The analysis was
then rerun unchanged and all results were retained.
