from ultralytics.utils.torch_utils import model_info_for_loggers

# Trainer callbacks ----------------------------------------------------------------------------------------------------
def on_pretrain_routine_start(trainer):
    """Called before the pretraining routine starts."""
    pass


def on_pretrain_routine_end(trainer):
    """Called after the pretraining routine ends."""
    pass


def on_train_start(trainer):
    """Called when the training starts."""
    print("[AIVDB] - [on_train_start]")
    import cv2 
    import os.path as osp
    import numpy as np
    
    for batch in trainer.train_loader:
        im_files = batch['im_file']
        imgs = batch['img']
        for im_file, img in zip(batch['im_file'], batch['img']):
            filename = osp.split(osp.splitext(im_file)[0])[-1]
            img = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))
            cv2.imwrite(osp.join(trainer.args.debug_dir, filename + '.jpg'), img)
        

def on_train_epoch_start(trainer):
    """Called at the start of each training epoch."""
    pass


def on_train_batch_start(trainer):
    """Called at the start of each training batch."""
    pass


def optimizer_step(trainer):
    """Called when the optimizer takes a step."""
    pass


def on_before_zero_grad(trainer):
    """Called before the gradients are set to zero."""
    pass


def on_train_batch_end(trainer):
    """Called at the end of each training batch."""
    pass


def on_train_epoch_end(trainer):
    """Called at the end of each training epoch."""
    print("[AIVDB] - [on_train_epoch_end]")
    """Log metrics and save images at the end of each training epoch."""
    print('- epoch: ', trainer.epoch + 1)
    print('- loss: ', trainer.label_loss_items(trainer.tloss, prefix="train"))
    print('- lr: ', trainer.lr)
    # if trainer.epoch == 1:
    #     _log_plots(trainer.plots, step=trainer.epoch + 1)
        
def on_fit_epoch_end(trainer):
    """Logs training metrics and model information at the end of an epoch."""
    print("[AIVDB] - [on_fit_epoch_end]")
    print("epoch: ", trainer.epoch + 1)
    print("- metrics: ", trainer.metrics)
    # _log_plots(trainer.plots, step=trainer.epoch + 1)
    # _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        print(model_info_for_loggers(trainer))

def on_model_save(trainer):
    print("[AIVDB] - [on_fit_epoch_end]")


def on_train_end(trainer):
    """Called when the training ends."""
    """Save the best model as an artifact at end of training."""
    print("[AIVDB] - [on_train_end]")
    # _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    # _log_plots(trainer.plots, step=trainer.epoch + 1)
    # art = wb.Artifact(type="model", name=f"run_{wb.run.id}_model")
    # if trainer.best.exists():
    #     art.add_file(trainer.best)
    #     wb.run.log_artifact(art, aliases=["best"])
    # # Check if we actually have plots to save
    # if trainer.args.plots and hasattr(trainer.validator.metrics, "curves_results"):
    #     for curve_name, curve_values in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
    #         x, y, x_title, y_title = curve_values
    #         _plot_curve(
    #             x,
    #             y,
    #             names=list(trainer.validator.metrics.names.values()),
    #             id=f"curves/{curve_name}",
    #             title=curve_name,
    #             x_title=x_title,
    #             y_title=y_title,
    #         )


def on_params_update(trainer):
    """Called when the model parameters are updated."""
    pass


def teardown(trainer):
    """Called during the teardown of the training process."""
    pass


# Validator callbacks --------------------------------------------------------------------------------------------------


def on_val_start(validator):
    """Called when the validation starts."""
    pass


def on_val_batch_start(validator):
    """Called at the start of each validation batch."""
    pass


def on_val_batch_end(validator):
    """Called at the end of each validation batch."""
    pass


def on_val_end(validator):
    """Called when the validation ends."""
    pass


# Predictor callbacks --------------------------------------------------------------------------------------------------


def on_predict_start(predictor):
    """Called when the prediction starts."""
    pass


def on_predict_batch_start(predictor):
    """Called at the start of each prediction batch."""
    pass


def on_predict_batch_end(predictor):
    """Called at the end of each prediction batch."""
    pass


def on_predict_postprocess_end(predictor):
    """Called after the post-processing of the prediction ends."""
    pass


def on_predict_end(predictor):
    """Called when the prediction ends."""
    pass


# Exporter callbacks ---------------------------------------------------------------------------------------------------


def on_export_start(exporter):
    """Called when the model export starts."""
    pass


def on_export_end(exporter):
    """Called when the model export ends."""
    pass


default_callbacks = {
    # Run in trainer
    "on_pretrain_routine_start": [on_pretrain_routine_start],
    "on_pretrain_routine_end": [on_pretrain_routine_end],
    "on_train_start": [on_train_start],
    "on_train_epoch_start": [on_train_epoch_start],
    "on_train_batch_start": [on_train_batch_start],
    "optimizer_step": [optimizer_step],
    "on_before_zero_grad": [on_before_zero_grad],
    "on_train_batch_end": [on_train_batch_end],
    "on_train_epoch_end": [on_train_epoch_end],
    "on_fit_epoch_end": [on_fit_epoch_end],  # fit = train + val
    "on_model_save": [on_model_save],
    "on_train_end": [on_train_end],
    "on_params_update": [on_params_update],
    "teardown": [teardown],
    # Run in validator
    "on_val_start": [on_val_start],
    "on_val_batch_start": [on_val_batch_start],
    "on_val_batch_end": [on_val_batch_end],
    "on_val_end": [on_val_end],
    # Run in predictor
    "on_predict_start": [on_predict_start],
    "on_predict_batch_start": [on_predict_batch_start],
    "on_predict_postprocess_end": [on_predict_postprocess_end],
    "on_predict_batch_end": [on_predict_batch_end],
    "on_predict_end": [on_predict_end],
    # Run in exporter
    "on_export_start": [on_export_start],
    "on_export_end": [on_export_end],
}
