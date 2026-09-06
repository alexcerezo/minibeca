#!/usr/bin/env bash
# provision_pod.sh — Crea (o reutiliza) un volumen y un pod GPU adecuados
# para entrenar el SAE en menos de una hora.
#
# Requiere:
#   - runpodctl instalado (curl -sSL https://cli.runpod.net | bash)
#   - RUNPOD_API_KEY exportada (o configurada vía `runpodctl config`)
#   - Una clave SSH pública registrada en https://console.runpod.io/user/settings
#
# Salida: imprime IDs de pod y volumen, y una línea `eval` para exportar
#         IP/PORT/KEY para SSH.

set -euo pipefail

: "${POD_NAME:=sae-gpt2}"
: "${VOL_NAME:=sae-vol}"
: "${VOL_SIZE_GB:=50}"
: "${DC:=EU-RO-1}"                 # DC con A100/L40S disponibles la mayor parte del tiempo
: "${GPU_ID:=NVIDIA A100 80GB PCIe}"   # o 'NVIDIA L40S'
: "${TEMPLATE_ID:=runpod-torch-v280}"  # torch 2.8.0 + CUDA 12.8 + Python 3.12 + Ubuntu 24.04
: "${TERMINATE_AFTER:=+3h}"        # cost guard: el pod se elimina solo pasadas 3h

command -v runpodctl >/dev/null || { echo "runpodctl no está instalado" >&2; exit 1; }
runpodctl user >/dev/null || { echo "RUNPOD_API_KEY inválida o ausente" >&2; exit 1; }

# 1) Volumen persistente (idempotente por nombre)
VOL_ID="$(runpodctl network-volume list --output json 2>/dev/null \
          | python3 -c "
import json, sys
name = '${VOL_NAME}'
for v in json.load(sys.stdin):
    if v.get('name') == name:
        print(v['id']); break
" || true)"
if [[ -z "${VOL_ID}" ]]; then
    echo "==> creando volumen ${VOL_NAME} (${VOL_SIZE_GB} GB) en ${DC}"
    VOL_ID="$(runpodctl network-volume create \
                --name "${VOL_NAME}" --size "${VOL_SIZE_GB}" \
                --data-center-id "${DC}" --output json | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')"
fi
echo "VOL_ID=${VOL_ID}"

# 2) Pod GPU con SSH, sin puertos http (batch job), volumen montado en /workspace
echo "==> creando pod ${POD_NAME} (${GPU_ID}, template ${TEMPLATE_ID}) en ${DC}"
POD_ID="$(runpodctl pod create --name "${POD_NAME}" \
            --template-id "${TEMPLATE_ID}" --gpu-id "${GPU_ID}" \
            --data-center-ids "${DC}" \
            --network-volume-id "${VOL_ID}" --volume-mount-path /workspace \
            --ssh --terminate-after "${TERMINATE_AFTER}" \
            --output json | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')"
echo "POD_ID=${POD_ID}"

# 3) Espera al runtime y extrae info SSH
echo "==> esperando runtime del pod (~30-90s)…"
for _ in $(seq 1 30); do
    STATE="$(runpodctl pod get "${POD_ID}" --output json 2>/dev/null \
             | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("runtime") is not None)')"
    [[ "${STATE}" == "True" ]] && break
    sleep 5
done

runpodctl ssh info "${POD_ID}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'export POD_ID={d[\"id\"]!s}')
print(f'export POD_IP={d[\"ip\"]!s}')
print(f'export POD_PORT={d[\"port\"]!s}')
print(f'export POD_KEY={d[\"ssh_key\"][\"path\"]!s}')
"
echo "# --- SSH one-liner ---"
echo "# ssh -i \$POD_KEY -o StrictHostKeyChecking=no -p \$POD_PORT root@\$POD_IP"
