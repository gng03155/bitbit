# 바이낸스 선물 자동매매 봇 설치 가이드

본 프로젝트는 Ubuntu 24.04 및 Python 3.12 환경에 최적화되어 있습니다.

## 1. 사전 준비

- 바이낸스 API 키 (Futures 권한 활성화)
- n8n Webhook URL (Telegram 알림 연동용)
- PM2 (프로세스 관리도구)

## 2. 가상 환경 및 라이브러리 설치

시스템 환경과 분리하여 독립된 환경을 구축합니다.

```bash
# 가상 환경 생성
python3 -m venv venv

# 가상 환경 활성화
source venv/bin/activate

# pip 최신화 및 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```
