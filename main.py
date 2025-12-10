import asyncio
import threading
import tkinter as tk

import Emisor
import Receptor
from visualizador import LiveScope


def launch_ui():
    root = tk.Tk()
    emit_q = Emisor.get_emit_queue()
    recv_q = Receptor.get_recv_queue()  # asegúrate de haberlo agregado en Receptor
    LiveScope(root, emit_q, recv_q, fs=50.0)
    root.mainloop()


async def run_all():
    # UI en hilo
    threading.Thread(target=launch_ui, daemon=True).start()

    # Servidor emisor
    server_task = asyncio.create_task(Emisor.main())

    # Espera corta para que el server quede arriba
    await asyncio.sleep(0.4)

    # Cliente receptor (esto activa handle_connection)
    recv_task = asyncio.create_task(Receptor.receive_message())

    await asyncio.gather(server_task, recv_task)


if __name__ == "__main__":
    asyncio.run(run_all())
