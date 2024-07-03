# # myproject/routing.py
# from django.urls import path
# from channels.routing import ProtocolTypeRouter, URLRouter
# from channels.auth import AuthMiddlewareStack
# from hello import views
#
# application = ProtocolTypeRouter({
#     "websocket": AuthMiddlewareStack(
#         URLRouter([
#             path("ws/realtime/", views.RealtimeConsumer.as_asgi()),
#         ])
#     ),
# })

