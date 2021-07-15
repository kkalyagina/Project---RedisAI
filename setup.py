import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="RedMask",
    version="0.0.1",
    author="Konstantin Sviblov, Kristina Kaliagina, Andrei Savchuk, Andrei Gavrilov",
    author_email="andrei_gavrilov@epam.com",
    description="RedisAI wrapper for MaskFace project",
    url="https://kb.epam.com/display/~Andrei_Gavrilov/Reids+AI+for+Covid+Masks",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=required
)
