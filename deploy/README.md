# Desplegar la API en EC2

Esto documenta cómo se armó la instancia real (ver `docs/arquitectura.md`, sección "Despliegue en
EC2"), para poder reproducirla si se borra o se crea una nueva.

## Instancia

- Amazon Linux 2023, `t3.small` (la cuenta solo permite tipos elegibles para Free Tier).
- Perfil de instancia `BlindnessDetection-EC2Role` (necesita además la política administrada
  `AmazonSSMManagedInstanceCore` para entrar con Session Manager).
- Security group: solo entrada en el puerto 8000 (la API), nada de SSH.
- Entrar con `aws ssm start-session --target <instance-id>`, no SSH.

## Instalar la app

```bash
sudo dnf install -y git
cd /home/ec2-user
git clone https://github.com/GualNavarr04/BlindnessDetection.git app
cd app
python3 -m venv .venv
# /tmp es tmpfs (RAM): con poca memoria, pip se queda sin espacio al bajar el wheel de PyTorch.
mkdir -p /home/ec2-user/tmp
TMPDIR=/home/ec2-user/tmp .venv/bin/pip install --upgrade pip
TMPDIR=/home/ec2-user/tmp .venv/bin/pip install -r requirements.txt
```

Con 2 GB de RAM, calcular Grad-CAM puede tronar el proceso (*OOM killer*). Agregar swap:

```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
```

## Servicio systemd

Copiar `blindness-api.service` (cambiando `<account-id>`) a `/etc/systemd/system/` y:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now blindness-api
```

## Logs a CloudWatch

```bash
sudo dnf install -y amazon-cloudwatch-agent
sudo mkdir -p /opt/aws/amazon-cloudwatch-agent/etc
sudo cp cloudwatch-agent-config.json /opt/aws/amazon-cloudwatch-agent/etc/config.json
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config -m ec2 -s -c file:/opt/aws/amazon-cloudwatch-agent/etc/config.json
```

No hace falta ningún permiso de IAM nuevo: `BlindnessDetection-EC2Role` ya tiene acceso al grupo
de logs `/blindness-detection/*`.

## Apagar cuando no se use

La instancia cobra por hora: `aws ec2 stop-instances --instance-ids <id>`. El servicio y el
agente quedan habilitados (`enable`), así que arrancan solos al volver a prender la instancia —
solo cambia la IP pública.
