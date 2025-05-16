# SHAP Microservice Render.com Deployment Checklist

## Pre-Deployment Checks

- [ ] Code changes committed to version control
- [ ] `requirements.txt` includes all necessary dependencies
- [ ] `render.yaml` updated with optimized settings
- [ ] Model files available or `create_minimal_model.py` ready
- [ ] Redis connection details confirmed and tested
- [ ] Memory optimization settings adjusted and tested locally

## Deployment Day Steps

1. [ ] **Preparation**
   - [ ] Run `./deploy_to_render_final.sh`
   - [ ] Verify `.skip_training` file is created
   - [ ] Confirm model files exist in `models/` directory
   - [ ] Check that optimized environment variables are set

2. [ ] **Deployment**
   - [ ] **Option A:** Git-based deployment
     - [ ] Commit all changes: `git add . && git commit -m "Deploy optimized SHAP microservice"`
     - [ ] Push to repository: `git push origin main`
     - [ ] Confirm Render.com triggered automatic deployment
   
   - [ ] **Option B:** Manual deployment
     - [ ] Log into Render.com dashboard
     - [ ] Navigate to your project
     - [ ] Click "Manual Deploy" then "Deploy latest commit"

3. [ ] **Monitoring Deployment**
   - [ ] Watch build logs for errors
   - [ ] Confirm all dependencies installed successfully
   - [ ] Verify pre-deployment checks passed
   - [ ] Check that both web service and worker start correctly

4. [ ] **Verification**
   - [ ] Run `./verify_render_deployment.sh`
   - [ ] Check service responds to HTTP requests
   - [ ] Verify Redis connection is working
   - [ ] Monitor memory usage (should be under 475MB)
   - [ ] Submit test job and confirm successful processing
   - [ ] Check worker logs for any errors

## Post-Deployment Tasks

- [ ] Document the deployed service URL
- [ ] Share API documentation with team members
- [ ] Set up monitoring alerts in Render dashboard
- [ ] Schedule regular health checks
- [ ] Update internal documentation with new settings

## Rollback Plan (If Needed)

1. [ ] **Identify Issue**
   - [ ] Check logs to pinpoint problem
   - [ ] Determine if it's configuration or code related

2. [ ] **Quick Fixes**
   - [ ] Try adjusting environment variables in Render dashboard
   - [ ] Restart services if needed

3. [ ] **Full Rollback**
   - [ ] Revert to previous commit
   - [ ] Redeploy using the same method as initial deployment
   - [ ] Verify rollback was successful

## Contact Information

- **Tech Lead:** [Name] ([Email])
- **DevOps Support:** [Name] ([Email])
- **Render.com Account Admin:** [Name] ([Email])
