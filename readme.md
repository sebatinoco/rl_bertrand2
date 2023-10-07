# Presentación

Hola! Este es el repositorio del proyecto de tesis *Colusión Algorítmica en Contexto Inflacionario* a presentar para obtener el grado de Magíster de Ciencia de Datos de la Universidad de Chile. 

# Abstract

Este proyecto tiene por objetivo añadir evidencia a la *Colusión Algorítmica*, fenómeno que ocurre cuando se usan algoritmos de Reinforcement Learning para el pricing en un mercado de oligopolio. 

El mercado es modelado a través del modelo de *Bertrand*, el cual intenta ilustrar la competencia a través de los precios y donde un menor precio siempre será preferido por los clientes. Si los agentes actuan de forma individual, lo anterior implica que el precio de equilibrio (*Precio de Nash*) se sitúe en niveles cercanos al costo por unidad de cada agente, pues siempre existirá un incentivo a reducir los precios para capturar un mayor nivel de demanda. 

¿Es esto lo que ocurre en el siguiente gráfico?

![Screenshot](plots/Bertrand/Bertrand_N-4_lr-0.5_k-1.png)

# Instrucciones de ejecución

Ejecutar este proyecto es simple, solo debes seguir los siguientes pasos:
1. Clonar este repositorio: `git clone https://github.com/sebatinoco/rl_bertrand`
2. Crear un nuevo ambiente virtual (se recomienda hacer esto a través de conda: `conda create --name new_env`)
3. Activar el ambiente creado: `conda activate new_env`
4. Instalar los paquetes necesarios para su ejecución: `pip install -r requirements.txt`
5. Finalmente, para obtener los resultados presentados: `bash run.sh`

No olvides revisar los parámetros disponibles en `utils.parse_args.py` para obtener más resultados interesantes!

*Cualquier duda o comentario no dudes en escribirme a stinoco@fen.uchile.cl*
# rl_bertrand2
