# Sign-project

## To start the game on windows operating system, follow these steps :

* Download python 3.10.11 version

* Clone the repository by running the following command in your terminal :

```
git clone https://github.com/Sign-Language-Group2/Sign-project.git
```

* Change to the project directory :

```
cd Sign-project/
```

* Open the project :

```
code .
```

* Open a new terminal (PowerShell) .

* Create a virtual environment by running the following command:

```
python -m venv .venv
```

```
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process
```


* Activate the virtual environment.

```
.venv\Scripts\activate
```

* Verify that the correct version of Python is installed by running:

```
python --version
```
It should display Python 3.10.11.


* Create an empty init file by running the following command:

```
touch __init__.py
```

* Change to the "Game" directory:

```
cd Game
```
* Create another empty init file inside the "Game" directory:

```
touch __init__.py
```
* Go back to the parent directory:

```
cd ..
```

* Install the required dependencies by running:

```
pip install -r requirements.txt
```
* Start the game by running the following command:

```
python .\Game\play.py
```