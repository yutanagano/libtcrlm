from libtcrlm.tokeniser.tokeniser import Tokeniser
from libtcrlm.tokeniser.token_indices import (
    AminoAcidTokenIndex,
    CdrCompartmentIndex,
    SingleChainCdrCompartmentIndex,
)
from libtcrlm.schema import Tcr
import torch
from torch import Tensor
from typing import List, Optional, Tuple


class CdrTokeniser(Tokeniser):
    """
    Tokenise TCR in terms of its alpha and beta chain CDRs 1 2 and 3.

    Dim 0: token index
    Dim 1: topen position
    Dim 2: CDR length
    Dim 3: CDR index
    """

    token_vocabulary_index = AminoAcidTokenIndex

    def tokenise(self, tcr: Tcr) -> Tensor:
        initial_cls_vector = (AminoAcidTokenIndex.CLS, 0, 0, CdrCompartmentIndex.NULL)

        cdr1a = self._convert_to_numerical_form(
            tcr.cdr1a_sequence, CdrCompartmentIndex.CDR1A
        )
        cdr2a = self._convert_to_numerical_form(
            tcr.cdr2a_sequence, CdrCompartmentIndex.CDR2A
        )
        cdr3a = self._convert_to_numerical_form(
            tcr.junction_a_sequence, CdrCompartmentIndex.CDR3A
        )

        cdr1b = self._convert_to_numerical_form(
            tcr.cdr1b_sequence, CdrCompartmentIndex.CDR1B
        )
        cdr2b = self._convert_to_numerical_form(
            tcr.cdr2b_sequence, CdrCompartmentIndex.CDR2B
        )
        cdr3b = self._convert_to_numerical_form(
            tcr.junction_b_sequence, CdrCompartmentIndex.CDR3B
        )

        all_cdrs_tokenised = (
            [initial_cls_vector] + cdr1a + cdr2a + cdr3a + cdr1b + cdr2b + cdr3b
        )

        number_of_tokens_other_than_initial_cls = len(all_cdrs_tokenised) - 1
        if number_of_tokens_other_than_initial_cls == 0:
            raise RuntimeError(f"tcr {tcr} does not contain any TCR information")

        return torch.tensor(all_cdrs_tokenised, dtype=torch.long)

    def _convert_to_numerical_form(
        self, aa_sequence: Optional[str], cdr_index: CdrCompartmentIndex
    ) -> List[Tuple[int]]:
        if aa_sequence is None:
            return []

        token_indices = [AminoAcidTokenIndex[aa] for aa in aa_sequence]
        token_positions = [idx for idx, _ in enumerate(aa_sequence, start=1)]
        cdr_length = [len(aa_sequence) for _ in aa_sequence]
        compartment_index = [cdr_index for _ in aa_sequence]

        iterator_over_token_vectors = zip(
            token_indices, token_positions, cdr_length, compartment_index
        )

        return list(iterator_over_token_vectors)


class AlphaCdrTokeniser(Tokeniser):
    """
    Tokenise TCR in terms of its alpha chain CDRs 1 2 and 3.

    Dim 0: token index
    Dim 1: topen position
    Dim 2: CDR length
    Dim 3: CDR index
    """

    token_vocabulary_index = AminoAcidTokenIndex

    def tokenise(self, tcr: Tcr) -> Tensor:
        initial_cls_vector = (
            AminoAcidTokenIndex.CLS,
            0,
            0,
            SingleChainCdrCompartmentIndex.NULL,
        )

        cdr1b = self._convert_to_numerical_form(
            tcr.cdr1a_sequence, SingleChainCdrCompartmentIndex.CDR1
        )
        cdr2b = self._convert_to_numerical_form(
            tcr.cdr2a_sequence, SingleChainCdrCompartmentIndex.CDR2
        )
        cdr3b = self._convert_to_numerical_form(
            tcr.junction_a_sequence, SingleChainCdrCompartmentIndex.CDR3
        )

        all_cdrs_tokenised = [initial_cls_vector] + cdr1b + cdr2b + cdr3b

        number_of_tokens_other_than_initial_cls = len(all_cdrs_tokenised) - 1
        if number_of_tokens_other_than_initial_cls == 0:
            raise RuntimeError(f"tcr {tcr} does not contain any TRA information")

        return torch.tensor(all_cdrs_tokenised, dtype=torch.long)

    def _convert_to_numerical_form(
        self, aa_sequence: Optional[str], cdr_index: SingleChainCdrCompartmentIndex
    ) -> List[Tuple[int]]:
        if aa_sequence is None:
            return []

        token_indices = [AminoAcidTokenIndex[aa] for aa in aa_sequence]
        token_positions = [idx for idx, _ in enumerate(aa_sequence, start=1)]
        cdr_length = [len(aa_sequence) for _ in aa_sequence]
        compartment_index = [cdr_index for _ in aa_sequence]

        iterator_over_token_vectors = zip(
            token_indices, token_positions, cdr_length, compartment_index
        )

        return list(iterator_over_token_vectors)


class BetaCdrTokeniser(Tokeniser):
    """
    Tokenise TCR in terms of its beta chain CDRs 1 2 and 3.

    Dim 0: token index
    Dim 1: topen position
    Dim 2: CDR length
    Dim 3: CDR index
    """

    token_vocabulary_index = AminoAcidTokenIndex

    def tokenise(self, tcr: Tcr) -> Tensor:
        initial_cls_vector = (
            AminoAcidTokenIndex.CLS,
            0,
            0,
            SingleChainCdrCompartmentIndex.NULL,
        )

        cdr1b = self._convert_to_numerical_form(
            tcr.cdr1b_sequence, SingleChainCdrCompartmentIndex.CDR1
        )
        cdr2b = self._convert_to_numerical_form(
            tcr.cdr2b_sequence, SingleChainCdrCompartmentIndex.CDR2
        )
        cdr3b = self._convert_to_numerical_form(
            tcr.junction_b_sequence, SingleChainCdrCompartmentIndex.CDR3
        )

        all_cdrs_tokenised = [initial_cls_vector] + cdr1b + cdr2b + cdr3b

        number_of_tokens_other_than_initial_cls = len(all_cdrs_tokenised) - 1
        if number_of_tokens_other_than_initial_cls == 0:
            raise RuntimeError(f"tcr {tcr} does not contain any TRB information")

        return torch.tensor(all_cdrs_tokenised, dtype=torch.long)

    def _convert_to_numerical_form(
        self, aa_sequence: Optional[str], cdr_index: SingleChainCdrCompartmentIndex
    ) -> List[Tuple[int]]:
        if aa_sequence is None:
            return []

        token_indices = [AminoAcidTokenIndex[aa] for aa in aa_sequence]
        token_positions = [idx for idx, _ in enumerate(aa_sequence, start=1)]
        cdr_length = [len(aa_sequence) for _ in aa_sequence]
        compartment_index = [cdr_index for _ in aa_sequence]

        iterator_over_token_vectors = zip(
            token_indices, token_positions, cdr_length, compartment_index
        )

        return list(iterator_over_token_vectors)
