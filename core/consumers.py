# core/consumers.py

import json

from channels.generic.websocket import AsyncWebsocketConsumer


class DetectionConsumer(AsyncWebsocketConsumer):
  async def connect(self):
    self.camera_numero = self.scope['url_route']['kwargs']['camera_numero']
    self.room_group_name = f'detect_{self.camera_numero}'

    # Unirse al grupo de la sala
    await self.channel_layer.group_add(
      self.room_group_name,
      self.channel_name
    )

    await self.accept()

  async def disconnect(self, close_code):
    # Salir del grupo de la sala
    await self.channel_layer.group_discard(
      self.room_group_name,
      self.channel_name
    )

  # Recibir mensaje desde el WebSocket (no lo usaremos en este caso)
  async def receive(self, text_data):
    pass

  # Recibir mensaje desde el grupo de la sala y enviarlo al cliente
  async def detection_update(self, event):
    message = event['message']

    # Enviar mensaje al WebSocket
    await self.send(text_data=json.dumps(message))
