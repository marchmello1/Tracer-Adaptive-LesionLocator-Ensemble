ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee
FROM ${BASE_IMAGE}

RUN groupadd --system algorithm && useradd --system --create-home --gid algorithm algorithm && \
    mkdir -p /opt/algorithm /opt/algorithm/public_champion/model \
      /opt/algorithm/edt_code /opt/algorithm/edt_model/fold_0 \
      /input /output/images/tumor-lesion-segmentation && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

COPY --chown=algorithm:algorithm requirements.txt constraints.txt /opt/algorithm/
COPY --chown=algorithm:algorithm autoPET-interactive /opt/algorithm/autoPET-interactive
RUN python -m pip install --user -r requirements.txt -c constraints.txt && \
    python -m pip install --user -e /opt/algorithm/autoPET-interactive -c constraints.txt && \
    python -m pip install --user edt==3.1.2 -c constraints.txt

COPY --chown=algorithm:algorithm champion/autopet-3-submission /opt/algorithm/public_champion/autopet-3-submission
COPY --chown=algorithm:algorithm champion/predict_wrap.py /opt/algorithm/public_champion/predict_wrap.py
COPY --chown=algorithm:algorithm autoPET-interactive/nnunetv2 /opt/algorithm/edt_code/nnunetv2
COPY --chown=algorithm:algorithm candidate_runtime /opt/algorithm/candidate_runtime
COPY --chown=algorithm:algorithm edt_runner.py public_tracer_router.py tracer_router_ensemble_v1.npz /opt/algorithm/
COPY --chown=algorithm:algorithm weights/edt_model /opt/algorithm/edt_model
COPY --chown=algorithm:algorithm download_weights.py /opt/algorithm/download_weights.py

USER root
ARG CHAMPION_WEIGHTS_URL=https://zenodo.org/api/records/14007247/files/autoPET-3-LesionTracer.zip/content
ARG CHAMPION_WEIGHTS_MD5=566016409b0bd14770c0b57c1f2873f1
ARG EDT_WEIGHTS_URL
ARG EDT_WEIGHTS_SHA256=a0cb3a89c72b0a79a27900980361385ff02572c0c71aba6609390fecbbc13e82
RUN set -eux; \
    python /opt/algorithm/download_weights.py "${CHAMPION_WEIGHTS_URL}" /tmp/champion.zip; \
    test "$(md5sum /tmp/champion.zip | cut -d' ' -f1)" = "${CHAMPION_WEIGHTS_MD5}"; \
    python -c "import zipfile; zipfile.ZipFile('/tmp/champion.zip').extractall('/opt/algorithm/public_champion/model')"; \
    rm /tmp/champion.zip; \
    test "$(find /opt/algorithm/public_champion/model -name checkpoint_final.pth | wc -l)" -eq 5; \
    python /opt/algorithm/download_weights.py "${EDT_WEIGHTS_URL}" /opt/algorithm/edt_model/fold_0/checkpoint_final.pth; \
    test "$(sha256sum /opt/algorithm/edt_model/fold_0/checkpoint_final.pth | cut -d' ' -f1)" = "${EDT_WEIGHTS_SHA256}"; \
    chown -R algorithm:algorithm /opt/algorithm/public_champion /opt/algorithm/edt_model

USER algorithm
ENV nnUNet_raw=/opt/algorithm/nnUNet_raw
ENV nnUNet_preprocessed=/opt/algorithm/nnUNet_preprocessed
ENV nnUNet_results=/opt/algorithm/public_champion/model
ENV nnUNet_compile=0
ENV AUTOPET_PUBLIC_TRACER_MODE=robust
ENV AUTOPET_FDG_RELAX_BURDEN_COMPONENTS=25
ENV AUTOPET_FDG_PROBABILITY_THRESHOLD=0.47
ENV AUTOPET_PSMA_PRUNE_THRESHOLD=0.86
ENV AUTOPET_EDT_CODE=/opt/algorithm/edt_code
ENV AUTOPET_EDT_MODEL=/opt/algorithm/edt_model
ENV AUTOPET_EDT_RUNNER=/opt/algorithm/edt_runner.py

ENTRYPOINT ["python", "-m", "candidate_runtime.process_rank2_final_interaction"]
