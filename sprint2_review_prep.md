# Review prep — Portugal Housing Price Prediction

Notes for the project review. Two audiences in the room: a product manager and a senior data scientist. Lead with the business story, then go deep on the technical choices when asked.

---

## 1. The 60-second pitch

Portugal Real Estate Brokers want to spot underpriced listings automatically. I built a regression model that predicts a listing's asking price from its features (location, area, type, energy certificate). When the model's predicted price sits well above the listed price, that listing is a candidate for the brokers to chase. The model is served as a containerized HTTP API so it can be called whenever a new listing lands in the system.

Headline numbers (test set):
- Best single deployable model: tuned **LightGBM**, RMSE ≈ €263k, R² ≈ 0.61
- Best overall: a **Voting ensemble**, RMSE ≈ €260k, R² ≈ 0.62 (only ~1.4% better than LightGBM)
- I deploy LightGBM, not the ensemble. See the deployment tradeoff below.

---

## 2. Key insights to highlight (business side)

- **Location dominates.** District is the single biggest price driver. Lisboa carries a large, statistically significant premium (t-test, p < 0.001).
- **Size is second.** Total area and living area are the next strongest predictors.
- **Energy efficiency has a real premium.** A+/A certificates ask materially more than C/D/NC (ANOVA, p < 0.001). Part efficiency, part correlation with newer builds.
- **The model is honest about its ceiling.** R² ≈ 0.62 means ~38% of price variance is unexplained, largely because asking price ≠ sale price and 12 columns were too sparse to use.

---

## 3. The four likely questions

### Q1. Bagging vs boosting

Both are ensembles of decision trees; the difference is how the trees are built and combined.

- **Bagging (Random Forest).** Trees are trained independently and in parallel, each on a bootstrap sample of the rows and a random subset of features. Predictions are averaged. The point is **variance reduction**: any single deep tree overfits, but averaging many decorrelated trees cancels out their individual errors. Hard to overfit by adding more trees.
- **Boosting (Gradient Boosting, XGBoost, LightGBM).** Trees are trained sequentially. Each new tree fits the residual errors of the ensemble so far, and its contribution is scaled by a learning rate. The point is **bias reduction** (with variance control via regularization): the model keeps correcting its own mistakes. More powerful, but can overfit if you add too many trees or set the learning rate too high.

One line: bagging averages independent trees to cut variance; boosting adds dependent trees to cut bias. On this data boosting and bagging landed very close, with boosting marginally ahead, which says the signal has enough interaction structure for sequential correction to help a little.

### Q2. What makes XGBoost / LightGBM fast and efficient

Both improve on textbook gradient boosting in similar ways:

- **Histogram-based splits.** Instead of scanning every continuous value for the best split, they bucket features into a fixed number of bins (e.g. 255) and evaluate splits on the bins. Far fewer candidate splits to check.
- **Regularization built in.** L1/L2 penalties on leaf weights plus a minimum-gain-to-split threshold, so trees don't grow useless branches.
- **Native missing-value handling.** Each split learns a default direction for NaNs, so no imputation is strictly required (this is also why XGBoost could train on raw NaNs while the linear models could not).
- **Parallelism and cache-aware data layout.** Split-finding is parallelized across features; data is stored column-wise in compressed blocks.

LightGBM specifics worth naming:
- **Leaf-wise growth** (grow the leaf with the largest loss reduction) instead of level-wise. Fewer leaves for the same accuracy, but needs `num_leaves`/`max_depth` control to avoid overfitting.
- **GOSS** (keep large-gradient samples, subsample small-gradient ones) and **EFB** (bundle mutually-exclusive sparse features). Both cut the effective data size with little accuracy loss, which is why LightGBM is usually the fastest on large, wide datasets.

XGBoost specifics: the **weighted quantile sketch** for approximate split-finding and a **sparsity-aware** algorithm for the default-direction trick.

### Q3. Key steps to deploy a model to production

1. **Serialize** the trained pipeline (preprocessing + model together) so inference matches training exactly. I used joblib.
2. **Wrap it in a service** with a defined API contract. FastAPI here: a `/predict` endpoint with a typed request schema (Pydantic) and a `/health` check.
3. **Containerize** with Docker so the runtime, dependencies, and model ship as one immutable image.
4. **Deploy to a host** that can serve HTTP and scale. Cloud Run here: serverless, scales to zero, pay per request.
5. **Test the live endpoint**, then add the operational layer: monitoring, logging, and a retraining/rollout plan (covered in the notebook's recommendations).

I'd also mention what I deliberately deferred: drift monitoring, a CI/CD retrain pipeline, and canary/blue-green rollout. Out of scope here, but I know where they slot in.

### Q4. Why Docker, and how it ensures consistency across environments

A model depends on exact library versions; scikit-learn, LightGBM, and even NumPy can change behavior between releases. "Works on my machine" happens when the training environment and the serving environment quietly differ.

Docker packages the code, the model file, the Python runtime, and every pinned dependency into one image defined by a Dockerfile. That image runs identically on my laptop, a teammate's machine, and Cloud Run, because it carries its own environment instead of borrowing the host's. So the container I tested locally is byte-for-byte the same one that serves in the cloud. That reproducibility is the whole point, and it's also what makes scaling and rollback safe.

---

## 4. Technical choices to be ready to defend

- **Why drop 12 columns?** They were >50% missing. Imputing a mostly-empty column injects more noise than signal. Documented in §4.
- **Why RMSE as the primary metric?** It's in euros (reads as "typical error in €") and penalizes large misses, which matches the business cost. MAE/R²/MAPE reported alongside.
- **Why deploy LightGBM, not the best model?** The Voting ensemble is ~1.4% better on RMSE but serializes to ~800 MB (a deep Random Forest inside it) and won't fit a 512 MB Cloud Run container. LightGBM is 2.45 MB, loads instantly, and gives up almost nothing. That is the right production call, and naming it shows engineering judgment.
- **Why keep all property types?** Filtering to residential-only would drop ~40% of rows and remove signal the model uses to separate, say, land from apartments. Trees handle the mix well.
- **Leakage control.** Preprocessing lives inside the sklearn Pipeline, so it is fit only on training folds during CV. The test set is sealed from §5 until §13.
- **Why SHAP on LightGBM and not XGBoost?** The installed XGBoost serializes `base_score` in a format the current SHAP can't parse. LightGBM is fully compatible, and using it keeps SHAP, feature importance, and the learning curve on one consistent model.

---

## 5. Honest limitations (say these before you're asked)

- Asking price is not sale price. The target is what sellers list, not what buyers pay.
- Scraped data: duplicates, entry errors, inconsistent formatting.
- R² ≈ 0.62 is moderate. Useful for ranking listings, not for a precise valuation.
- No temporal features: interest rates, seasonality, and market movement are absent.
- City/town dropped for cardinality; target encoding could recover finer location signal.
