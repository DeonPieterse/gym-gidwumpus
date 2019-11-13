from setuptools import setup

setup(
  name = 'gym_gidwumpus',         # How you named your package folder (MyLib)
  packages = ['gym_gidwumpus'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'This is an environment for the wumpus world that you can reference as an openai gym environment.',   # Give a short description about your library
  author = 'Gideon Jacobus Pieterse',                   # Type in your name
  author_email = 'Gideondeonjacobus@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/Gid8/gym-gidwumpus',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/Gid8/gym-gidwumpus/archive/0.1.tar.gz',    # I explain this later on
  keywords = ['openai', 'ai', 'reinforcement', 'learning', 'reinforcement learning', 'wimpus', 'env', 'environment', 'machine', 'machine learning'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'gym',
          'numpy',
          'tqdm'
      ],
  classifiers=[
    'Development Status :: Production',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
