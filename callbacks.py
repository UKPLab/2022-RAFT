from transformers_modified import TrainerCallback
from transformers_modified.integrations import WandbCallback
from rational.torch import Rational
from args import CustomTrainingArguments
import tempfile
import numbers
from pathlib import Path
import torch.nn.utils.prune as prune
import os


class RationalEvoGraph(TrainerCallback):
    def __init__(self, approx_func) -> None:
        self.approx_func = approx_func
        super().__init__()
    def on_epoch_end(self, args, state, control, **kwargs):
        Rational.capture_all(name = f"step {state.global_step}")
    # def on_step_end(self, args, state, control, **kwargs):
    #     Rational.capture_all(name = f"step {state.global_step}")
    def on_train_end(self, args, state, control, **kwargs):
        Rational.export_evolution_graphs(path = f"./logs/images/{self.approx_func}_RF_evalution.gif")


class CustomWandbCallback(WandbCallback):

    def __init__(self):
        super().__init__()

        self.pruned_image_dir = os.path.join('./logs/images', 'pruning')


        self.normal_image_dir = os.path.join('./logs/images','normal')
        if not os.path.exists(self.pruned_image_dir):
            os.makedirs(self.pruned_image_dir)

        if not os.path.exists(self.normal_image_dir):
            os.makedirs(self.normal_image_dir)

    # def on_step_begin(
    #     self, args:CustomTrainingArguments, state, control, **kwargs
    # ):
    #     if args.save_rational_plots:
    #         if state.global_step % args.logging_steps == 0:
    #             Rational.save_all_inputs(True)
    # def on_step_end(self, args: CustomTrainingArguments, state, control, **kwargs):
    #     if state.global_step % args.save_steps==0:
    #         print('true......')

    #         image_dir  = os.path.join(self.normal_image_dir, args.run_name)
    #         if not os.path.exists(image_dir):
    #             os.mkdir(image_dir)
    #         filename = os.path.join(image_dir, f"{args.approx_func}_RF_step_{state.global_step}.jpg")
    #         Rational.capture_all(name = "RF")
    #         Rational.export_graphs(path = filename)

    def on_epoch_end(
        self, args:CustomTrainingArguments, state, control, **kwargs
    ):
        if args.save_rational_plots and len(Rational.list) > 0:
            Rational.capture_all(f"Global Step {state.global_step}")
            # filename = f"{args.output_dir}/ra_{state.global_step}.png"
            # Rational.export_graphs(filename)
            # self._wandb.log(
                # {
                    # "train/rational_activations": wandb.Image(filename),
                    # "train/global_step": state.global_step,
                # }
            # )

    def on_evaluate(self, args: CustomTrainingArguments, state, control, **kwargs):

        if args.save_rational_plots:
            image_dir  = os.path.join(self.normal_image_dir, args.run_name)
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)
            filename = os.path.join(image_dir, f"{args.approx_func}_RF.jpg")
            Rational.capture_all(name = "func")
            Rational.export_graphs(path = filename)


    def on_train_end(self, args:CustomTrainingArguments, state, control, model=None, tokenizer=None, **kwargs):
        if args.save_rational_plots:
            # if state.global_step % args.logging_steps == 0 and len(Rational.list) > 0:
            if args.do_pruning:
                image_dir  = os.path.join(self.pruned_image_dir, args.run_name)
            else:
                image_dir = os.path.join(self.normal_image_dir, args.run_name)
                
            filename = os.path.join(image_dir, f"{args.approx_func}_RF_evalution.gif")
            try:
                Rational.export_evolution_graphs(path = filename)
            except:
                Rational.export_graphs(path = filename)
            self._wandb.log(
                        {
                            "train/rational_activations": self._wandb.Image(filename),
                            "train/global_step": state.global_step,
                        }
                    )
        if self._log_model and self._initialized and state.is_world_process_zero:
            from transformers_modified import Trainer

            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                    }
                )
                artifact = self._wandb.Artifact(name=f"model-{self._wandb.run.id}", type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)

