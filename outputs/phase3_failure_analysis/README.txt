Phase 3.1 safety failure analysis for the existing 30-image SegFormer evaluation.
Error map colors:
- dark gray: correct prediction
- magenta: GT high risk predicted low risk, critical unsafe error
- orange: GT high risk predicted medium risk, caution error
- cyan: GT low risk predicted high risk, conservative false alarm
- white: other disagreement

Maximum softmax probability is a confidence proxy only, not calibrated uncertainty.
