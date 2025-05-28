import os
import sys
import subprocess
import time

def test_startup():
    print("🔍 Testing application startup...")
    
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    render_deploy_dir = os.path.join(project_root, 'render_deploy')
    
    # Set up environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{project_root}:{render_deploy_dir}"
    env['PORT'] = '10000'
    
    # Change to render_deploy directory
    os.chdir(render_deploy_dir)
    
    # Start gunicorn in a subprocess
    cmd = [
        'gunicorn',
        '--config', 'gunicorn_config.py',
        'app:app',
        '--bind', '0.0.0.0:10000'
    ]
    
    print(f"📝 Running command: {' '.join(cmd)}")
    print(f"📝 PYTHONPATH: {env['PYTHONPATH']}")
    print(f"📝 Current directory: {os.getcwd()}")
    
    try:
        # Start gunicorn
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Give it a few seconds to start up
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Application started successfully!")
            
            # Get the output
            stdout, stderr = process.communicate()
            print("\n📋 Output:")
            print(stdout)
            if stderr:
                print("\n⚠️ Errors:")
                print(stderr)
        else:
            print("❌ Application failed to start!")
            stdout, stderr = process.communicate()
            print("\n⚠️ Errors:")
            print(stderr)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        # Clean up
        if process.poll() is None:
            process.terminate()
            process.wait()

if __name__ == '__main__':
    test_startup() 