# agent/tools/pagos.py
from langchain_core.tools import tool

@tool
def calcular_monto_bcv(monto_usd: float):
    """Calcula el monto en bolívares usando la tasa del BCV."""
    print(f"--- Herramienta: Calculando {monto_usd} USD a tasa BCV ---")
    tasa_bcv = 150.00  # Simulado
    return f"El monto en bolívares es {monto_usd * tasa_bcv} Bs."

@tool
def verificar_pago(id_comprobante: str):
    """Verifica un comprobante de pago en el sistema."""
    print(f"--- Herramienta: Verificando pago con ID {id_comprobante} ---")
    return "Pago verificado exitosamente."