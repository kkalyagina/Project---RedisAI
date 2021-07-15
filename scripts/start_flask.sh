export PATH=/etc/crontab:$PATH && \
crontab -l 2>/dev/null > update_model && \
echo "00 01 * * 1-5 python ./scripts/update_model.py" >> update_model && \
crontab update_model && \
rm update_model && \
python ./scripts/flask_service.py
