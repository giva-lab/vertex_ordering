## Project Repository Template

Use this as a foundation to kickstart your project repository. To get started, simply click on the green button labeled 'Use this template'.

![Repository Image](https://github.com/giva-lab/template/assets/30710465/67e8f0af-25b6-4c26-8bbd-2a601a53f001)

### Getting Started

1. **Clone Repository:** Clone this repository onto your local machine.
2. **Install Poetry:** Ensure you have Poetry installed on your system.
3. **Setup Environment:** Open a terminal in the project's root directory and execute the following commands:
    ```bash
    poetry shell
    poetry update
    pre-commit install
    ```
4. **Ready to Go!** You're all set to start working on the project!

### Additional Notes

- **Adding Packages:** You can add new packages via Poetry using the command `poetry add {package name}`. Please **DO NOT** add packages via pip, as it may lead to version conflicts in the future.
- **Commit Changes:** After adding a new package, remember to commit the modified `pyproject.toml` file so that other project members can access the new packages.
